"""
arXiv 论文抓取模块

功能:
1. 调用 arXiv API 获取论文元数据
2. 下载 PDF 文件
3. 解析论文信息
4. 增量更新抓取

运行示例:
    python -m paper_system.backend.arxiv_crawler --category cs --max-results 100
"""

import os
import time
import json
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import arxiv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# arXiv 分类目录
ARXIV_CATEGORIES = {
    "physics": [
        "astro-ph",
        "cond-mat",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math-ph",
        "nlin",
        "nucl-ex",
        "nucl-th",
        "physics",
        "quant-gas",
        "math",
    ],
    "cs": [
        "cs.AI",
        "cs.AR",
        "cs.CC",
        "cs.CE",
        "cs.CG",
        "cs.CL",
        "cs.CR",
        "cs.CV",
        "cs.CY",
        "cs.DB",
        "cs.DC",
        "cs.DL",
        "cs.DM",
        "cs.DS",
        "cs.ET",
        "cs.FL",
        "cs.GL",
        "cs.GR",
        "cs.GT",
        "cs.HC",
        "cs.IR",
        "cs.IT",
        "cs.LG",
        "cs.LO",
        "cs.MA",
        "cs.MM",
        "cs.MS",
        "cs.NA",
        "cs.NE",
        "cs.NI",
        "cs.OH",
        "cs.OS",
        "cs.PF",
        "cs.PL",
        "cs.RO",
        "cs.SC",
        "cs.SD",
        "cs.SE",
        "cs.SI",
        "cs.SY",
    ],
    "q-bio": [
        "q-bio.BM",
        "q-bio.CB",
        "q-bio.GN",
        "q-bio.MN",
        "q-bio.NC",
        "q-bio.OT",
        "q-bio.PE",
        "q-bio.QM",
        "q-bio.SC",
        "q-bio.TO",
    ],
    "q-fin": [
        "q-fin.CP",
        "q-fin.EC",
        "q-fin.GN",
        "q-fin.MF",
        "q-fin.PM",
        "q-fin.PR",
        "q-fin.RM",
        "q-fin.ST",
        "q-fin.TR",
    ],
    "stat": ["stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH"],
    "math": [
        "math.AG",
        "math.AT",
        "math.AP",
        "math.ACT",
        "math.CT",
        "math.CA",
        "math.CO",
        "math.AC",
        "math.CV",
        "math.DG",
        "math.DS",
        "math.FA",
        "math.GM",
        "math.GN",
        "math.GT",
        "math.GR",
        "math.HO",
        "math.IT",
        "math.KT",
        "math.LO",
        "math.MP",
        "math.MG",
        "math.NT",
        "math.NA",
        "math.OA",
        "math.OC",
        "math.PR",
        "math.QA",
        "math.RT",
        "math.RA",
        "math.SP",
        "math.ST",
        "math.SG",
    ],
    "eess": ["eess.AS", "eess.IV", "eess.SP", "eess.SY"],
    "econ": ["econ.EM", "econ.GN", "econ.TH"],
    "eess": ["eess.AS", "eess.IV", "eess.SP", "eess.SY"],
    "all": [],  # 全部类别
}

# 全部类别列表
ALL_CATEGORIES = []
for cats in ARXIV_CATEGORIES.values():
    ALL_CATEGORIES.extend(cats)


@dataclass
class Paper:
    """论文数据模型"""

    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: str
    updated_date: str
    pdf_url: str
    doi: Optional[str] = None
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    download_status: str = "pending"
    local_pdf_path: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def display_id(self) -> str:
        return self.arxiv_id.replace(":", "_")


class ArxivCrawler:
    """arXiv 论文抓取器"""

    def __init__(
        self,
        data_dir: str = "e:/量子架构/paper_system/data",
        categories: Optional[List[str]] = None,
        max_results_per_category: int = 1000,
    ):
        self.data_dir = Path(data_dir)
        self.categories = categories or ALL_CATEGORIES
        self.max_results = max_results_per_category

        # 创建目录
        self.papers_dir = self.data_dir / "papers"
        self.papers_dir.mkdir(parents=True, exist_ok=True)

        self.client = arxiv.Client()

        # 加载已存在的论文索引
        self.index_file = self.data_dir / "paper_index.json"
        self.papers: Dict[str, Paper] = {}
        self._load_index()

    def _load_index(self):
        """加载论文索引"""
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.papers = {k: Paper(**v) for k, v in data.items()}
            logger.info(f"已加载 {len(self.papers)} 篇论文索引")

    def _save_index(self):
        """保存论文索引"""
        data = {k: v.to_dict() for k, v in self.papers.items()}
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def search_papers(
        self,
        query: str = "all",
        categories: Optional[List[str]] = None,
        max_results: int = 100,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[Paper]:
        """搜索论文

        Args:
            query: 搜索关键词
            categories: 指定分类
            max_results: 最大结果数
            sort_by: 排序方式 (submittedDate, relevance, lastUpdatedDate)
            sort_order: 升序/降序
        """
        cats = categories or self.categories
        cat_query = " OR ".join([f"cat:{c}" for c in cats[:10]])  # 限制类别数量

        if query and query != "all":
            search_query = f"all:{query} AND ({cat_query})"
        else:
            search_query = cat_query

        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=(
                arxiv.SortCriterion.SubmittedDate
                if sort_by == "submittedDate"
                else (
                    arxiv.SortCriterion.Relevance
                    if sort_by == "relevance"
                    else arxiv.SortCriterion.LastUpdatedDate
                )
            ),
            sort_order=(
                arxiv.SortOrder.Descending
                if sort_order == "descending"
                else arxiv.SortOrder.Ascending
            ),
        )

        papers = []
        for result in self.client.results(search):
            try:
                paper = Paper(
                    arxiv_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary.replace("\n", " "),
                    categories=result.categories,
                    published_date=str(result.published.date()) if result.published else "",
                    updated_date=str(result.updated.date()) if result.updated else "",
                    pdf_url=result.pdf_url,
                    doi=result.doi,
                    comment=result.comment,
                    journal_ref=result.journal_ref,
                )
                papers.append(paper)
            except Exception as e:
                logger.warning(f"解析论文失败: {e}")
                continue

        return papers

    def crawl_category(
        self,
        category: str,
        max_results: int = 1000,
        incremental: bool = True,
    ) -> List[Paper]:
        """抓取指定分类的论文

        Args:
            category: arXiv 分类
            max_results: 最大结果数
            incremental: 是否增量抓取
        """
        logger.info(f"开始抓取分类: {category}")

        # 构建查询
        query = f"cat:{category}"

        # 增量模式：只抓取新论文
        if incremental and self.papers:
            latest_date = max(
                (
                    datetime.fromisoformat(p.published_date)
                    for p in self.papers.values()
                    if p.published_date
                ),
                default=datetime(2020, 1, 1),
            )
            query += f" AND submittedDate:[{latest_date.strftime('%Y%m%d')} TO 99991231]"

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        new_papers = []
        for result in self.client.results(search):
            try:
                arxiv_id = result.entry_id.split("/")[-1]

                # 跳过已存在的
                if arxiv_id in self.papers:
                    continue

                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary.replace("\n", " "),
                    categories=result.categories,
                    published_date=str(result.published.date()) if result.published else "",
                    updated_date=str(result.updated.date()) if result.updated else "",
                    pdf_url=result.pdf_url,
                    doi=result.doi,
                    comment=result.comment,
                    journal_ref=result.journal_ref,
                )

                new_papers.append(paper)
                self.papers[arxiv_id] = paper

                logger.info(f"抓取: {paper.title[:50]}...")

                # 避免过于频繁请求
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"抓取失败: {e}")
                continue

        # 保存索引
        self._save_index()

        logger.info(f"分类 {category} 抓取完成，新增 {len(new_papers)} 篇论文")
        return new_papers

    def crawl_all(self, max_per_category: int = 500) -> List[Paper]:
        """抓取所有分类的论文"""
        all_papers = []

        for category in ALL_CATEGORIES:
            try:
                papers = self.crawl_category(category, max_per_category)
                all_papers.extend(papers)
            except Exception as e:
                logger.error(f"分类 {category} 抓取失败: {e}")
                continue

        return all_papers

    def download_pdf(self, paper: Paper, force: bool = False) -> str:
        """下载论文 PDF

        Args:
            paper: 论文对象
            force: 是否强制重新下载
        Returns:
            PDF 本地路径
        """
        # 检查是否已下载
        pdf_path = self.papers_dir / f"{paper.display_id}.pdf"

        if pdf_path.exists() and not force:
            paper.download_status = "completed"
            paper.local_pdf_path = str(pdf_path)
            return str(pdf_path)

        try:
            # 使用 arxiv 下载
            download = arxiv.Downloader(
                client=self.client,
                dir=self.papers_dir,
                filename="{index}.{extension}",
            )

            result = self.client.results(arxiv.Search(f"id:{paper.arxiv_id}")).__iter__().__next__()

            # 下载
            downloaded_path = download.download(result)

            paper.download_status = "completed"
            paper.local_pdf_path = str(pdf_path)
            self._save_index()

            logger.info(f"下载完成: {paper.title[:30]}")
            return str(pdf_path)

        except Exception as e:
            paper.download_status = f"failed: {e}"
            logger.warning(f"下载失败: {paper.arxiv_id} - {e}")
            return ""

    def batch_download(self, max_count: int = 100) -> List[str]:
        """批量下载论文"""
        pending = [p for p in self.papers.values() if p.download_status != "completed"]

        downloaded = []
        for paper in pending[:max_count]:
            path = self.download_pdf(paper)
            if path:
                downloaded.append(path)
            time.sleep(1)  # 避免过于频繁

        return downloaded

    def get_statistics(self) -> Dict:
        """获取抓取统计"""
        total = len(self.papers)
        downloaded = sum(1 for p in self.papers.values() if p.download_status == "completed")

        # 分类统计
        category_counts = {}
        for paper in self.papers.values():
            for cat in paper.categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_papers": total,
            "downloaded": downloaded,
            "pending": total - downloaded,
            "category_counts": category_counts,
        }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="arXiv 论文抓取工具")
    parser.add_argument(
        "--category", "-c", type=str, default="cs", help="分类 (cs, physics, math, all)"
    )
    parser.add_argument("--max-results", "-m", type=int, default=100, help="最大结果数")
    parser.add_argument("--download", "-d", action="store_true", help="下载 PDF")
    parser.add_argument("--query", "-q", type=str, default="", help="搜索关键词")

    args = parser.parse_args()

    # 确定分类
    if args.category == "all":
        categories = ALL_CATEGORIES
    else:
        categories = ARXIV_CATEGORIES.get(args.category, [args.category])

    # 创建抓取器
    crawler = ArxivCrawler()

    # 搜索论文
    if args.query:
        papers = crawler.search_papers(args.query, categories, args.max_results)
        print(f"找到 {len(papers)} 篇论文")
        for p in papers[:10]:
            print(f"  - {p.title[:60]}")
    else:
        # 抓取分类
        for cat in categories[:5]:  # 限制数量
            papers = crawler.crawl_category(cat, args.max_results)
            print(f"分类 {cat}: 新增 {len(papers)} 篇")

    # 下载 PDF
    if args.download:
        crawler.batch_download(10)

    # 打印统计
    stats = crawler.get_statistics()
    print(f"\n统计: 总计 {stats['total_papers']} 篇, 已下载 {stats['downloaded']} 篇")


if __name__ == "__main__":
    main()
