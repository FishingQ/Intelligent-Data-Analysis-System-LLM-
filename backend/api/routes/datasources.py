"""
数据源管理接口
POST /api/datasources/upload —— 上传文件
GET /api/datasources —— 列出已注册数据源
DELETE /api/datasources/{source_id} —— 移除数据源
"""
import os
import uuid
import shutil
import logging
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from backend.shared.schemas import (
    DataSourceConfig, DataSourceType, DataSourceUploadResponse,
    TableSchema,
)
from backend.data_sources.factory import DataSourceFactory

logger = logging.getLogger(__name__)

router = APIRouter()

# 上传目录
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))),
    "data", "uploads"
)

# 数据源注册表 (source_id → DataSourceConfig)
# 注意: 这里和 chat.py 共享 _source_registry?
# 简单方案: 放在模块级别，两个路由都可以 import
_source_registry: dict = {}


def get_source_registry() -> dict:
    """获取数据源注册表（供 chat.py 使用）"""
    return _source_registry


# ---- 文件扩展名 → 数据源类型映射 ----
EXT_TYPE_MAP = {
    "xlsx": DataSourceType.EXCEL,
    "xls": DataSourceType.EXCEL,
    "csv": DataSourceType.CSV,
    "sqlite": DataSourceType.SQLITE,
    "db": DataSourceType.SQLITE,
}


@router.post("/datasources/upload", response_model=DataSourceUploadResponse)
async def upload_datasource(file: UploadFile = File(...)):
    """
    上传数据文件（Excel / CSV / SQLite）

    处理流程:
    1. 校验文件扩展名
    2. 保存到 data/uploads/
    3. 创建适配器 → 解析 Schema
    4. 返回 source_id + 表结构预览
    """
    # 1. 文件名校验
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in EXT_TYPE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: .{ext}，支持: {list(EXT_TYPE_MAP.keys())}",
        )

    source_type = EXT_TYPE_MAP[ext]

    # 2. 保存文件
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_id = str(uuid.uuid4())[:12]
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file.filename)
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"文件已保存: {save_path} ({os.path.getsize(save_path)} bytes)")

    # 3. 创建适配器并解析 Schema
    config = DataSourceConfig(
        source_type=source_type,
        source_id=file_id,
        display_name=file.filename,
        connection_params={"file_path": save_path},
    )

    try:
        adapter = DataSourceFactory.create(config)
        adapter.connect()
        tables = adapter.get_all_schemas()
        adapter.disconnect()
    except Exception as e:
        logger.error(f"解析数据源失败: {e}")
        # 即使解析失败也注册，让用户知道文件已上传
        tables = []
        # 但如果是致命错误（文件损坏），则删除文件
        if "不存在" in str(e) or "损坏" in str(e) or "corrupt" in str(e).lower():
            os.remove(save_path)
            raise HTTPException(status_code=400, detail=f"文件解析失败: {e}")

    # 4. 注册到全局注册表
    _source_registry[file_id] = config

    # 同时注册到 chat 路由的注册表（通过 import）
    from backend.api.routes.chat import register_datasource
    register_datasource(config)

    logger.info(
        f"数据源注册成功: {file.filename} → {file_id} "
        f"({len(tables)}个表)"
    )

    return DataSourceUploadResponse(
        source_id=file_id,
        display_name=file.filename,
        source_type=source_type.value,
        filepath=save_path,
        tables=tables,
    )


@router.get("/datasources")
async def list_datasources():
    """列出所有已注册的数据源"""
    sources = []
    for sid, cfg in _source_registry.items():
        adapter = DataSourceFactory.get_adapter(sid)
        sources.append({
            "source_id": sid,
            "display_name": cfg.display_name,
            "source_type": cfg.source_type.value,
            "connected": adapter.is_connected if adapter else False,
            "file_path": cfg.connection_params.get("file_path", ""),
        })
    return {"sources": sources, "count": len(sources)}


@router.get("/datasources/{source_id}/schema")
async def get_datasource_schema(source_id: str):
    """获取指定数据源的表结构"""
    cfg = _source_registry.get(source_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"数据源不存在: {source_id}")

    try:
        adapter = DataSourceFactory.get_or_create(cfg)
        tables = adapter.get_all_schemas()
        # 转为可 JSON 序列化的格式
        return {
            "source_id": source_id,
            "display_name": cfg.display_name,
            "tables": [t.model_dump() for t in tables],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取Schema失败: {e}")


@router.delete("/datasources/{source_id}")
async def remove_datasource(source_id: str):
    """移除数据源（断开连接 + 可选删除文件）"""
    if source_id not in _source_registry:
        raise HTTPException(status_code=404, detail=f"数据源不存在: {source_id}")

    cfg = _source_registry.pop(source_id)
    DataSourceFactory.remove_adapter(source_id)

    logger.info(f"数据源已移除: {cfg.display_name} ({source_id})")
    return {"status": "ok", "message": f"数据源 '{cfg.display_name}' 已移除"}
