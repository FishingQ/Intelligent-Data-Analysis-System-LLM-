"""
CSV 数据源适配器
使用 DuckDB 让 CSV 文件支持 SQL 查询
"""
import os
import time
from typing import List

import pandas as pd

from backend.shared.schemas import QueryResult, TableSchema, ColumnInfo
from .base import BaseAdapter


class CSVAdapter(BaseAdapter):
    """CSV 文件适配器 —— 文件本身作为一张表，表名为文件名(去掉.csv)"""

    def connect(self) -> bool:
        path = self.config.connection_params.get("file_path", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV 文件不存在: {path}")
        self._file_path = path
        # 尝试多种编码读取
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']:
            try:
                self._df = pd.read_csv(path, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            self._df = pd.read_csv(path, encoding='utf-8', errors='replace')

        # 清理列名
        self._df.columns = [str(c).strip() for c in self._df.columns]
        self._connected = True
        return True

    def get_tables(self) -> List[str]:
        self._ensure_connected()
        # 表名为文件名(无扩展名)
        name = os.path.splitext(os.path.basename(self._file_path))[0]
        # 清理为安全的表名（字母/数字/下划线）
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
        # 以数字开头时加前缀 t_ (SQL标识符不能以数字开头)
        if safe_name and safe_name[0].isdigit():
            safe_name = "t_" + safe_name
        return [safe_name]

    def get_schema(self, table_name: str) -> TableSchema:
        self._ensure_connected()
        df = self._df
        columns = []
        for col_name in df.columns:
            dtype_str = str(df[col_name].dtype)
            sample_series = df[col_name].dropna()
            samples = sample_series.head(3).tolist() if len(sample_series) > 0 else []

            samples_clean = []
            for s in samples:
                try:
                    if hasattr(s, 'item'):
                        samples_clean.append(s.item())
                    elif isinstance(s, (pd.Timestamp,)):
                        samples_clean.append(str(s))
                    else:
                        samples_clean.append(s)
                except Exception:
                    samples_clean.append(str(s))

            columns.append(ColumnInfo(
                name=str(col_name),
                data_type=dtype_str,
                nullable=bool(df[col_name].isna().any()),
                sample_values=samples_clean,
            ))

        return TableSchema(
            table_name=table_name,
            columns=columns,
            row_count=len(df),
            description="",
        )

    def execute(self, sql: str) -> QueryResult:
        """使用 DuckDB 执行 SQL 查询"""
        self._ensure_connected()
        import duckdb

        start = time.time()
        try:
            con = duckdb.connect()
            table_name = self.get_tables()[0]
            con.register(table_name, self._df)

            result_df = con.execute(sql).df()
            con.close()

            elapsed_ms = (time.time() - start) * 1000
            return QueryResult(
                success=True,
                columns=list(result_df.columns),
                data=result_df.values.tolist(),
                row_count=len(result_df),
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
        self._df = None
        self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            raise RuntimeError("未连接到 CSV 文件，请先调用 connect()")
