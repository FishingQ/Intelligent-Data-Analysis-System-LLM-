"""
SQLAlchemy 共用适配器基类 (M3-3 + M3-4)
MySQL / PostgreSQL 适配器共用逻辑：连接、Schema内省、SQL执行
"""
import time
import logging
from typing import List
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from backend.shared.schemas import QueryResult, TableSchema, ColumnInfo, DataSourceType
from .base import BaseAdapter

logger = logging.getLogger(__name__)

# Provider → SQLAlchemy 连接串前缀
DIALECT_MAP = {
    DataSourceType.MYSQL: "mysql+pymysql",
    DataSourceType.POSTGRESQL: "postgresql+psycopg2",
}


class SQLAlchemyAdapter(BaseAdapter):
    """MySQL / PostgreSQL 共用适配器基类

    子类只需指定 source_type，其余逻辑全部复用:
        class MySQLAdapter(SQLAlchemyAdapter):
            pass  # source_type 在 DataSourceConfig 中指定
    """

    def connect(self) -> bool:
        params = self.config.connection_params
        dialect = DIALECT_MAP.get(self.config.source_type)

        if not dialect:
            raise ValueError(f"不支持的数据源类型: {self.config.source_type}")

        # 构建连接 URL: dialect://user:password@host:port/database
        user = params.get("user", "")
        password = quote_plus(params.get("password", ""))
        host = params.get("host", "localhost")
        port = params.get("port", 3306 if self.config.source_type == DataSourceType.MYSQL else 5432)
        database = params.get("database", "")

        url = f"{dialect}://{user}:{password}@{host}:{port}/{database}"

        try:
            self._engine = create_engine(
                url,
                pool_size=5,
                pool_recycle=3600,
                connect_args={"connect_timeout": 10} if self.config.source_type == DataSourceType.MYSQL else {},
            )
            # 测试连接
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._inspector = inspect(self._engine)
            self._connected = True
            logger.info(f"数据库连接成功: {host}:{port}/{database}")
            return True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"数据库连接失败: {e}")

    def get_tables(self) -> List[str]:
        self._ensure_connected()
        return self._inspector.get_table_names()

    def get_schema(self, table_name: str) -> TableSchema:
        self._ensure_connected()

        # 列信息
        cols = self._inspector.get_columns(table_name)
        # 主键
        pk = self._inspector.get_pk_constraint(table_name)
        pk_cols = set(pk.get("constrained_columns", [])) if pk else set()
        # 外键
        fks = self._inspector.get_foreign_keys(table_name)
        fk_map = {}
        for fk in fks:
            constrained = fk.get("constrained_columns", [])
            referred = fk.get("referred_table", "")
            for cc in constrained:
                fk_map[cc] = referred

        columns = []
        for col in cols:
            col_name = col["name"]
            columns.append(ColumnInfo(
                name=col_name,
                data_type=str(col.get("type", "")),
                nullable=col.get("nullable", True),
                is_primary_key=col_name in pk_cols,
                is_foreign_key=col_name in fk_map,
                referenced_table=fk_map.get(col_name),
            ))

        # 行数
        try:
            row_count = pd.read_sql(f"SELECT COUNT(*) AS cnt FROM {table_name}", self._engine).iloc[0, 0]
            row_count = int(row_count)
        except Exception:
            row_count = 0

        return TableSchema(
            table_name=table_name,
            columns=columns,
            row_count=row_count,
        )

    def execute(self, sql: str) -> QueryResult:
        self._ensure_connected()
        start = time.time()
        try:
            df = pd.read_sql(sql, self._engine)
            elapsed_ms = (time.time() - start) * 1000
            return QueryResult(
                success=True,
                columns=list(df.columns),
                data=df.values.tolist(),
                row_count=len(df),
                execution_time_ms=round(elapsed_ms, 2),
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return QueryResult(
                success=False, columns=[], data=[],
                row_count=0, execution_time_ms=round(elapsed_ms, 2),
                error_message=str(e),
            )

    def disconnect(self):
        if hasattr(self, '_engine') and self._engine:
            self._engine.dispose()
        self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            raise RuntimeError("未连接到数据库，请先调用 connect()")


class MySQLAdapter(SQLAlchemyAdapter):
    """MySQL 适配器 —— 继承 SQLAlchemyAdapter"""
    pass


class PostgreSQLAdapter(SQLAlchemyAdapter):
    """PostgreSQL 适配器 —— 继承 SQLAlchemyAdapter"""
    pass
