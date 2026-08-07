"""
数据源适配器基类
所有数据源适配器继承此类，实现统一的连接/查询/元数据接口
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

from backend.shared.schemas import TableSchema, QueryResult, ColumnInfo, DataSourceConfig


class BaseAdapter(ABC):
    """所有数据源适配器的抽象基类"""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """建立连接，返回是否成功"""
        ...

    @abstractmethod
    def get_tables(self) -> List[str]:
        """获取所有表名/Sheet名"""
        ...

    @abstractmethod
    def get_schema(self, table_name: str) -> TableSchema:
        """获取单个表的 Schema 信息"""
        ...

    @abstractmethod
    def execute(self, sql: str) -> QueryResult:
        """执行 SQL 查询并返回标准化结果"""
        ...

    def get_all_schemas(self) -> List[TableSchema]:
        """批量获取所有表的 Schema"""
        schemas = []
        for t in self.get_tables():
            try:
                schemas.append(self.get_schema(t))
            except Exception as e:
                print(f"[WARN] 获取表 {t} 的Schema失败: {e}")
        return schemas

    def preview(self, table_name: str, n: int = 5) -> Optional[pd.DataFrame]:
        """预览表的前N行数据"""
        result = self.execute(f"SELECT * FROM \"{table_name}\" LIMIT {n}")
        if result.success:
            return pd.DataFrame(result.data, columns=result.columns)
        return None

    def disconnect(self):
        """关闭连接"""
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
