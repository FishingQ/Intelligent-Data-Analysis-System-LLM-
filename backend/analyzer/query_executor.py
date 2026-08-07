"""
查询执行器
接收模块2生成的 GeneratedQuery，调用模块3的适配器执行

是模块2（智能解析）和模块3（数据引擎）之间的调度器
"""
import logging
import time
from typing import Dict, Optional

import pandas as pd

from backend.shared.schemas import GeneratedQuery, QueryResult
from backend.data_sources.base import BaseAdapter

logger = logging.getLogger(__name__)


class QueryExecutor:
    """
    查询执行器

    三种执行模式:
    - sql:    直接交给数据源适配器执行 SQL
    - python: 在受限沙箱中执行 Python 分析代码
    - hybrid: 先 SQL 获取数据，再 Python 二次处理

    用法:
        executor = QueryExecutor()
        result = executor.execute(generated_query, adapter)
    """

    # Python 沙箱允许的模块
    ALLOWED_IMPORTS = {
        "pandas": "pd",
        "numpy": "np",
    }

    def execute(
        self,
        query: GeneratedQuery,
        adapter: BaseAdapter,
    ) -> QueryResult:
        """
        执行查询

        Args:
            query: 模块2生成的查询对象
            adapter: 模块3的数据源适配器实例

        Returns:
            标准化的查询结果
        """
        if not query.code.strip():
            return QueryResult(
                success=False,
                error_message="生成的查询代码为空，请重新提问。",
            )

        logger.info(
            f"执行查询 | 类型: {query.query_type} | "
            f"目标: {query.target_table or 'auto'} | "
            f"代码长度: {len(query.code)}字符"
        )

        if query.query_type == "sql":
            return self._execute_sql(query, adapter)
        elif query.query_type == "python":
            return self._execute_python(query, adapter)
        elif query.query_type == "hybrid":
            return self._execute_hybrid(query, adapter)
        else:
            return QueryResult(
                success=False,
                error_message=f"不支持的查询类型: {query.query_type}",
            )

    def _execute_sql(
        self,
        query: GeneratedQuery,
        adapter: BaseAdapter,
    ) -> QueryResult:
        """执行纯 SQL 查询"""
        logger.debug(f"执行SQL: {query.code[:200]}...")
        result = adapter.execute(query.code)
        if result.success:
            logger.info(f"SQL执行成功: {result.row_count}行, {result.execution_time_ms:.0f}ms")
        else:
            logger.error(f"SQL执行失败: {result.error_message}")
        return result

    def _execute_python(
        self,
        query: GeneratedQuery,
        adapter: BaseAdapter,
    ) -> QueryResult:
        """
        在受限沙箱中执行 Python 代码

        约定: 代码中必须定义 result_df 变量作为最终输出
        """
        start = time.time()
        try:
            namespace = {
                "adapter": adapter,
                "pd": pd,
                "np": __import__("numpy"),
            }

            # 执行用户/LLM 提供的 Python 代码
            exec(query.code, namespace)

            # 获取结果
            result_df = namespace.get("result_df")
            if result_df is None:
                # 尝试从局部变量中找 DataFrame
                for var_name, var in namespace.items():
                    if isinstance(var, pd.DataFrame) and var_name != "adapter":
                        result_df = var
                        break

            if result_df is None:
                return QueryResult(
                    success=False,
                    error_message="Python代码执行后未找到 result_df 变量或 DataFrame 结果。",
                    execution_time_ms=(time.time() - start) * 1000,
                )

            # 确保是 DataFrame
            if not isinstance(result_df, pd.DataFrame):
                result_df = pd.DataFrame(result_df)

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
            logger.error(f"Python代码执行失败: {e}")
            return QueryResult(
                success=False,
                error_message=f"Python代码执行错误: {e}",
                execution_time_ms=round(elapsed_ms, 2),
            )

    def _execute_hybrid(
        self,
        query: GeneratedQuery,
        adapter: BaseAdapter,
    ) -> QueryResult:
        """
        混合执行: 先 SQL 再 Python

        代码约定: 使用变量 sql_result (dict) 获取 SQL 查询结果
                 必须定义 result_df 作为最终输出
        """
        start = time.time()
        try:
            namespace = {
                "adapter": adapter,
                "pd": pd,
                "np": __import__("numpy"),
                "sql_result": None,
            }

            exec(query.code, namespace)

            result_df = namespace.get("result_df")
            if result_df is None:
                for var_name, var in namespace.items():
                    if isinstance(var, pd.DataFrame) and var_name not in ("adapter",):
                        result_df = var
                        break

            if result_df is None:
                return QueryResult(
                    success=False,
                    error_message="Hybrid代码执行后未找到有效的结果DataFrame。",
                    execution_time_ms=(time.time() - start) * 1000,
                )

            if not isinstance(result_df, pd.DataFrame):
                result_df = pd.DataFrame(result_df)

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
            logger.error(f"Hybrid执行失败: {e}")
            return QueryResult(
                success=False,
                error_message=f"混合执行错误: {e}",
                execution_time_ms=round(elapsed_ms, 2),
            )
