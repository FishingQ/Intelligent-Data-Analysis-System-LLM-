"""
SQLite 数据源适配器
支持 .sqlite / .db 文件
"""
import sqlite3
import os
import time
from typing import List

from backend.shared.schemas import QueryResult, TableSchema, ColumnInfo
from .base import BaseAdapter


class SQLiteAdapter(BaseAdapter):
    """SQLite 文件适配器"""

    def connect(self) -> bool:
        path = self.config.connection_params.get("file_path", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"SQLite 文件不存在: {path}")
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._connected = True
        return True

    def get_tables(self) -> List[str]:
        self._ensure_connected()
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_schema(self, table_name: str) -> TableSchema:
        self._ensure_connected()
        # 获取列信息
        cursor = self._conn.execute(f"PRAGMA table_info(\"{table_name}\")")
        columns = []
        pk_cols = []
        for row in cursor.fetchall():
            columns.append(ColumnInfo(
                name=row[1],
                data_type=row[2],
                nullable=not bool(row[3]),
                is_primary_key=bool(row[5]),
            ))
            if bool(row[5]):
                pk_cols.append(row[1])

        # 获取外键
        fk_cursor = self._conn.execute(f"PRAGMA foreign_key_list(\"{table_name}\")")
        for fk_row in fk_cursor.fetchall():
            col_name = fk_row[3]
            for col in columns:
                if col.name == col_name:
                    col.is_foreign_key = True
                    col.referenced_table = fk_row[2]

        # 行数
        count = self._conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]

        return TableSchema(
            table_name=table_name,
            columns=columns,
            row_count=count,
            description="",
        )

    def execute(self, sql: str) -> QueryResult:
        self._ensure_connected()
        start = time.time()
        try:
            cursor = self._conn.execute(sql)
            col_names = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = [list(row) for row in cursor.fetchall()]
            elapsed_ms = (time.time() - start) * 1000
            return QueryResult(
                success=True,
                columns=col_names,
                data=rows,
                row_count=len(rows),
                execution_time_ms=round(elapsed_ms, 2),
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return QueryResult(
                success=False,
                columns=[],
                data=[],
                row_count=0,
                execution_time_ms=round(elapsed_ms, 2),
                error_message=str(e),
            )

    def disconnect(self):
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
        self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            raise RuntimeError("未连接到 SQLite 数据库，请先调用 connect()")
