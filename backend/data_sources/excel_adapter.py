"""
Excel 数据源适配器
使用 DuckDB 让 Excel 文件支持 SQL 查询
每个 Sheet 映射为一张"表"
"""
import os
import re
import time
from typing import List, Dict

import pandas as pd

from backend.shared.schemas import QueryResult, TableSchema, ColumnInfo
from .base import BaseAdapter


# 表头区域常见的"标题/注意事项"行关键字（整行稀疏时用于识别并跳过）
NOTE_KEYWORDS = ("注意", "须知", "填表", "提示")


def _not_empty(v) -> bool:
    """单元格是否有实际内容（排除 NaN / 空串 / 仅空白）"""
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    return str(v).strip() != ""


def _looks_like_data(row) -> bool:
    """判断一行是否像数据行（含数字/长ID/日期），用于区分表头与数据"""
    for v in row:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and not (isinstance(v, float) and pd.isna(v)):
            return True
        s = str(v).strip()
        if not s:
            continue
        if s.isdigit() and len(s) >= 6:
            return True
        if re.match(r"^\d{4}年\d{1,2}月\d{1,2}日", s):
            return True
    return False


def _clean_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """
    清洗从 header=None 读到的原始 Sheet：
    1. 跳过顶部稀疏的标题行 / 注意事项行，定位真实表头
    2. 若表头跨多行（多级表头），每列取最后一个非空标签（叶级）
    3. 去重列名，返回仅含数据、列名正确的 DataFrame
    """
    ncols = raw.shape[1]
    if ncols == 0:
        return raw
    limit = min(20, len(raw))

    # 1) 定位表头起始行：跳过顶部稀疏的标题/注意事项行
    counts = [
        sum(1 for v in raw.iloc[i].tolist() if _not_empty(v))
        for i in range(limit)
    ]
    header_start = 0
    for i in range(limit):
        row = raw.iloc[i].tolist()
        filled = [v for v in row if _not_empty(v)]
        text = "".join(str(v) for v in filled)
        # 稀疏行（合并单元格的标题/注意事项，通常只有 1 个值）
        is_sparse = counts[i] <= 1
        # 含"注意/须知/填表/提示"等关键字且相对列数很稀疏的说明行
        is_note = any(k in text for k in NOTE_KEYWORDS) and counts[i] < max(3, ncols // 2)
        later_denser = (i + 1 < limit) and max(counts[i + 1:]) > counts[i]
        if (is_sparse or is_note) and later_denser:
            continue
        header_start = i
        break

    # 2) 收集表头块：从 header_start 起，直到遇到数据行或空行
    header_rows = []
    data_start = header_start
    for i in range(header_start, limit):
        row = raw.iloc[i].tolist()
        if not any(_not_empty(v) for v in row):
            data_start = i + 1
            break
        if _looks_like_data(row):
            data_start = i
            break
        header_rows.append(row)
        data_start = i + 1

    # 3) 每列取表头块中最后一个非空标签
    header = []
    for j in range(ncols):
        label = None
        for row in header_rows:
            if j < len(row) and _not_empty(row[j]):
                label = str(row[j]).strip()
        header.append(label if label else f"column_{j}")

    # 4) 去重列名（DuckDB 注册要求唯一）
    seen: Dict[str, int] = {}
    for idx, h in enumerate(header):
        if h in seen:
            seen[h] += 1
            header[idx] = f"{h}_{seen[h]}"
        else:
            seen[h] = 0

    df = raw.iloc[data_start:].reset_index(drop=True)
    df.columns = header
    return df


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
            raw = pd.read_excel(self._file, sheet_name=sheet, header=None)
            df = _clean_sheet(raw)
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


if __name__ == "__main__":
    # 自检：模拟带"注意事项"行 + 多级表头的真实表格结构
    raw1 = pd.DataFrame([
        ["注意事项：本表数据仅供参考", None, None],
        ["学号", "姓名", None],
        [None, None, "备注"],
        [1001, "张三", "正常"],
        [1002, "李四", "缺失"],
    ])
    df1 = _clean_sheet(raw1)
    assert list(df1.columns) == ["学号", "姓名", "备注"], df1.columns
    assert df1.shape == (2, 3) and df1.iloc[0].tolist() == [1001, "张三", "正常"]

    # 无标题行时：表头在第一行，保持不变
    raw2 = pd.DataFrame([["a", "b"], [1, 2], [3, 4]])
    assert list(_clean_sheet(raw2).columns) == ["a", "b"]

    print("excel_adapter 自检通过")

