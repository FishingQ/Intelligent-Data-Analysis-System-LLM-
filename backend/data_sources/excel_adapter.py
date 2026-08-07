"""
Excel 数据源适配器
使用 DuckDB 让 Excel 文件支持 SQL 查询
每个 Sheet 映射为一张"表"
"""
import os
import time
from typing import List, Dict

import pandas as pd

from backend.shared.schemas import QueryResult, TableSchema, ColumnInfo
from .base import BaseAdapter


class ExcelAdapter(BaseAdapter):
    """Excel 文件适配器 (.xlsx / .xls) —— 每个 Sheet = 一张表"""

    def connect(self) -> bool:
        path = self.config.connection_params.get("file_path", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Excel 文件不存在: {path}")
        self._file_path = path
        self._file = pd.ExcelFile(path)
        self._sheets: Dict[str, pd.DataFrame] = {}           # safe_name → DataFrame
        self._name_map: Dict[str, str] = {}                   # safe_name → original_sheet_name
        for sheet in self._file.sheet_names:
            df = pd.read_excel(self._file, sheet_name=sheet)
            df.columns = [str(c).strip() for c in df.columns]
            safe = self._safe_name(sheet)
            self._sheets[safe] = df
            self._name_map[safe] = sheet
        self._connected = True
        return True

    def get_tables(self) -> List[str]:
        self._ensure_connected()
        return list(self._sheets.keys())

    def get_schema(self, table_name: str) -> TableSchema:
        self._ensure_connected()
        df = self._get_sheet(table_name)
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
        """使用 DuckDB 执行 SQL 查询（表名使用安全名称）"""
        self._ensure_connected()
        import duckdb

        start = time.time()
        try:
            con = duckdb.connect()
            for name, df in self._sheets.items():
                con.register(name, df)

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
                success=False, columns=[], data=[],
                row_count=0, execution_time_ms=round(elapsed_ms, 2),
                error_message=str(e),
            )

    def disconnect(self):
        self._sheets.clear()
        self._name_map.clear()
        self._connected = False

    def _get_sheet(self, name: str) -> pd.DataFrame:
        if name in self._sheets:
            return self._sheets[name]
        raise ValueError(f"Sheet '{name}' 不存在")

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = "".join(c if c.isalnum() or c == '_' else '_' for c in str(name))
        if safe and safe[0].isdigit():
            safe = "t_" + safe
        return safe

    def _ensure_connected(self):
        if not self._connected:
            raise RuntimeError("未连接到 Excel 文件，请先调用 connect()")
