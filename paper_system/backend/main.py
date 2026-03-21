"""
论文学习系统 - FastAPI 后端

功能:
1. 论文搜索和浏览
2. 论文元数据管理
3. 抓取任务管理
4. 论文学习进度跟踪
5. 知识图谱构建

启动: uvicorn paper_system.backend.main:app --reload
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

# 导入抓取模块
from arxiv_crawler import ArxivCrawler, Paper, ARXIV_CATEGORIES

# ============================================================================
# 配置
# ============================================================================

DATA_DIR = Path("e:/量子架构/paper_system/data")
PAPERS_DIR = DATA_DIR / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="论文学习系统 API",
    description="arXiv 论文抓取和学习系统",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化抓取器
crawler = ArxivCrawler(
    data_dir=str(DATA_DIR),
    categories=[],
    max_results_per_category=1000
)


# ============================================================================
# 数据模型
# ============================================================================

class PaperOut(BaseModel):
    """论文输出模型"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: str
    updated_date: str
    pdf_url: str
    doi: Optional[str] = None
    download_status: str = "pending"
    local_pdf_path: Optional[str] = None
    tags: List[str] = []
    read_status: str = "unread"  # unread, reading, completed
    notes: str = ""


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = ""
    categories: List[str] = []
    max_results: int = 20
    sort_by: str = "submittedDate"


class CrawlRequest(BaseModel):
    """抓取请求"""
    categories: List[str] = []
    max_per_category: int = 100
    download_pdfs: bool = False


class PaperUpdate(BaseModel):
    """论文更新"""
    read_status: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


# ============================================================================
# 论文管理 API
# ============================================================================

@app.get("/")
async def root():
    """返回前端页面"""
    return FileResponse("e:/量子架构/paper_system/frontend/index.html")


@app.get("/api/ping")
async def ping():
    """健康检查"""
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.get("/api/papers", response_model=List[PaperOut])
async def get_papers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    read_status: Optional[str] = None,
    search: Optional[str] = None,
):
    """获取论文列表"""
    papers = list(crawler.papers.values())
    
    # 过滤
    if category:
        papers = [p for p in papers if category in p.categories]
    
    if read_status:
        papers = [p for p in papers if getattr(p, 'read_status', 'unread') == read_status]
    
    if search:
        search = search.lower()
        papers = [p for p in papers 
                  if search in p.title.lower() or search in p.abstract.lower()]
    
    # 排序
    papers.sort(key=lambda x: x.published_date, reverse=True)
    
    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    
    return papers[start:end]


@app.get("/api/papers/{arxiv_id}", response_model=PaperOut)
async def get_paper(arxiv_id: str):
    """获取单篇论文详情"""
    paper = crawler.papers.get(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    return paper


@app.patch("/api/papers/{arxiv_id}")
async def update_paper(arxiv_id: str, update: PaperUpdate):
    """更新论文状态"""
    paper = crawler.papers.get(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    
    if update.read_status is not None:
        paper.read_status = update.read_status
    if update.tags is not None:
        paper.tags = update.tags
    if update.notes is not None:
        paper.notes = update.notes
    
    crawler._save_index()
    
    return {"status": "ok", "paper": paper}


# ============================================================================
# 搜索 API
# ============================================================================

@app.post("/api/search")
async def search_papers(req: SearchRequest):
    """搜索论文"""
    papers = crawler.search_papers(
        query=req.query,
        categories=req.categories,
        max_results=req.max_results,
    )
    
    return {
        "total": len(papers),
        "papers": papers
    }


# ============================================================================
# 抓取 API
# ============================================================================

@app.post("/api/crawl")
async def start_crawl(
    req: CrawlRequest,
    background_tasks: BackgroundTasks,
):
    """开始抓取论文"""
    def crawl_task(categories: List[str], max_count: int, download: bool):
        if not categories:
            categories = list(ARXIV_CATEGORIES.keys())[:3]  # 默认抓取前3个领域
        
        for cat in categories:
            try:
                if cat in ARXIV_CATEGORIES:
                    cats = ARXIV_CATEGORIES[cat]
                    for c in cats[:5]:  # 每个领域最多5个子类
                        crawler.crawl_category(c, max_count)
                else:
                    crawler.crawl_category(cat, max_count)
            except Exception as e:
                print(f"抓取失败: {cat} - {e}")
        
        if download:
            crawler.batch_download(50)
    
    background_tasks.add_task(crawl_task, req.categories, req.max_per_category, req.download_pdfs)
    
    return {"status": "started", "message": "抓取任务已启动"}


@app.get("/api/crawl/status")
async def get_crawl_status():
    """获取抓取状态"""
    stats = crawler.get_statistics()
    return stats


@app.get("/api/categories")
async def get_categories():
    """获取可用分类"""
    return ARXIV_CATEGORIES


# ============================================================================
# 统计 API
# ============================================================================

@app.get("/api/statistics")
async def get_statistics():
    """获取系统统计"""
    stats = crawler.get_statistics()
    
    # 添加阅读统计
    total = len(crawler.papers)
    read = sum(1 for p in crawler.papers.values() 
               if getattr(p, 'read_status', 'unread') != 'unread')
    
    stats['read_count'] = read
    stats['unread_count'] = total - read
    
    return stats


# ============================================================================
# 知识图谱 API
# ============================================================================

@app.get("/api/knowledge-graph")
async def get_knowledge_graph():
    """获取知识图谱数据"""
    # 构建简单的作者-论文关系图
    authors = {}
    links = []
    
    for paper in list(crawler.papers.values())[:500]:  # 限制数量
        for author in paper.authors[:5]:  # 每篇论文最多5个作者
            if author not in authors:
                authors[author] = {
                    'id': author,
                    'name': author,
                    'paper_count': 0,
                    'categories': set()
                }
            authors[author]['paper_count'] += 1
            authors[author]['categories'].update(paper.categories)
    
    # 转换为列表
    nodes = [
        {
            'id': name,
            'name': data['name'],
            'paper_count': data['paper_count'],
            'categories': list(data['categories'])[:3]
        }
        for name, data in authors.items()
    ]
    
    # 按论文数量排序，取前100
    nodes.sort(key=lambda x: x['paper_count'], reverse=True)
    nodes = nodes[:100]
    
    return {
        'nodes': nodes,
        'links': links,
        'total_authors': len(authors)
    }


# ============================================================================
# PDF 下载 API
# ============================================================================

@app.post("/api/download/{arxiv_id}")
async def download_paper(arxiv_id: str, background_tasks: BackgroundTasks):
    """下载论文 PDF"""
    paper = crawler.papers.get(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    
    def download_task():
        crawler.download_pdf(paper)
    
    background_tasks.add_task(download_task)
    
    return {"status": "started", "arxiv_id": arxiv_id}


@app.get("/api/pdf/{arxiv_id}")
async def get_pdf(arxiv_id: str):
    """获取 PDF 路径"""
    paper = crawler.papers.get(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    
    if paper.local_pdf_path and os.path.exists(paper.local_pdf_path):
        return {"path": paper.local_pdf_path, "url": f"/api/pdf/file/{arxiv_id}"}
    
    return {"path": None, "url": paper.pdf_url}


@app.get("/api/pdf/file/{arxiv_id}")
async def serve_pdf(arxiv_id: str):
    """服务 PDF 文件"""
    paper = crawler.papers.get(arxiv_id)
    if not paper or not paper.local_pdf_path:
        raise HTTPException(status_code=404, detail="PDF 不存在")
    
    return FileResponse(
        paper.local_pdf_path,
        media_type='application/pdf',
        headers={"Content-Disposition": f"inline; filename={arxiv_id}.pdf"}
    )


# ============================================================================
# 启动
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
