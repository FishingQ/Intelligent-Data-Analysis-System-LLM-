"""
数据源工厂
根据 DataSourceConfig 动态创建对应的适配器实例
"""
import importlib
import logging
from typing import List, Optional, Dict

from backend.shared.schemas import DataSourceType, DataSourceConfig, TableSchema
from .base import BaseAdapter

logger = logging.getLogger(__name__)

# 适配器注册表: source_type → (module_path, class_name)
ADAPTER_REGISTRY: Dict[DataSourceType, tuple] = {
    DataSourceType.SQLITE: ("backend.data_sources.sqlite_adapter", "SQLiteAdapter"),
    DataSourceType.EXCEL: ("backend.data_sources.excel_adapter", "ExcelAdapter"),
    DataSourceType.CSV: ("backend.data_sources.csv_adapter", "CSVAdapter"),
    # Phase 2: MySQL / PostgreSQL
    DataSourceType.MYSQL: ("backend.data_sources.sqlalchemy_adapter", "MySQLAdapter"),
    DataSourceType.POSTGRESQL: ("backend.data_sources.sqlalchemy_adapter", "PostgreSQLAdapter"),
}


class DataSourceFactory:
    """根据配置创建对应的适配器实例"""

    # 已连接的适配器缓存 (source_id → adapter)
    _instances: Dict[str, BaseAdapter] = {}

    @classmethod
    def create(cls, config: DataSourceConfig) -> BaseAdapter:
        """
        根据 DataSourceConfig 创建适配器实例

        Args:
            config: 数据源配置

        Returns:
            BaseAdapter 子类实例

        Raises:
            ValueError: 不支持的数据源类型
        """
        entry = ADAPTER_REGISTRY.get(config.source_type)
        if not entry:
            raise ValueError(
                f"不支持的数据源类型: {config.source_type.value}"
            )

        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"无法加载适配器 {config.source_type.value}: {e}"
            )

        return adapter_class(config)

    @classmethod
    def get_or_create(cls, config: DataSourceConfig) -> BaseAdapter:
        """获取已缓存的适配器，或创建新实例并缓存"""
        if config.source_id in cls._instances:
            adapter = cls._instances[config.source_id]
            if adapter.is_connected:
                return adapter

        adapter = cls.create(config)
        adapter.connect()
        cls._instances[config.source_id] = adapter
        return adapter

    @classmethod
    def get_adapter(cls, source_id: str) -> Optional[BaseAdapter]:
        """获取已缓存的适配器"""
        return cls._instances.get(source_id)

    @classmethod
    def remove_adapter(cls, source_id: str):
        """移除并断开缓存的适配器"""
        adapter = cls._instances.pop(source_id, None)
        if adapter:
            try:
                adapter.disconnect()
            except Exception:
                pass

    @classmethod
    def get_active_schemas_from_configs(
        cls, configs: List[DataSourceConfig]
    ) -> List[TableSchema]:
        """
        获取所有已激活数据源的 Schema

        Args:
            configs: 数据源配置列表

        Returns:
            所有表的 TableSchema 列表
        """
        all_schemas = []
        for cfg in configs:
            try:
                adapter = cls.get_or_create(cfg)
                schemas = adapter.get_all_schemas()
                all_schemas.extend(schemas)
                logger.info(
                    f"数据源 [{cfg.display_name}] 已连接: "
                    f"{len(schemas)} 个表"
                )
            except Exception as e:
                logger.warning(
                    f"数据源 [{cfg.display_name}] 连接失败: {e}"
                )
        return all_schemas

    @classmethod
    def clear_all(cls):
        """断开并清除所有缓存的适配器"""
        for sid in list(cls._instances.keys()):
            cls.remove_adapter(sid)
