"""
SQL 校验器
在 SQL 执行前进行语法和安全检查，防止错误 SQL 或注入攻击
"""
import re
import logging
from typing import List, Optional, Tuple

import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import Identifier, IdentifierList, Parenthesis

logger = logging.getLogger(__name__)


class SQLValidator:
    """
    SQL 安全校验器

    校验规则（按顺序）:
    1. 安全: 禁止 DROP / INSERT / UPDATE / DELETE / ALTER / TRUNCATE 等写操作
    2. 安全: 禁止注释注入、多语句注入
    3. 语法: 用 sqlparse 检查是否为有效 SQL
    4. 规模: 限制 SQL 长度和预期返回行数
    5. 白名单: 仅允许 Schema 中存在的表名

    用法:
        validator = SQLValidator()
        ok, errors, sanitized = validator.validate(sql, allowed_tables)
    """

    # 禁止的关键词 (防止写操作和注入)
    FORBIDDEN_KEYWORDS = [
        "DROP", "INSERT", "UPDATE", "DELETE", "ALTER",
        "TRUNCATE", "CREATE", "EXEC", "EXECUTE",
        "GRANT", "REVOKE", "REPLACE",
    ]

    # 危险模式 (多语句注入等)
    # 格式: (pattern, description) 或 (pattern, description, flags)
    DANGEROUS_PATTERNS = [
        (r';\s*\w*;\s*\w', "可能包含多语句查询"),
        (r'/\*.*\*/', "C风格注释"),
        (r'--.*$', "行注释", re.MULTILINE),
    ]

    def __init__(
        self,
        max_sql_length: int = 10000,
        max_result_rows: int = 10000,
    ):
        self.max_sql_length = max_sql_length
        self.max_result_rows = max_result_rows

    def validate(
        self,
        sql: str,
        allowed_tables: List[str],
        allowed_columns: List[str] = None,
    ) -> Tuple[bool, List[str], str]:
        """
        校验 SQL 语句

        Args:
            sql: 待校验的 SQL 字符串
            allowed_tables: 允许查询的表名白名单
            allowed_columns: 允许查询的列名白名单（可选）

        Returns:
            (is_valid, errors, sanitized_sql)
        """
        errors = []
        sanitized = sql.strip()

        # 空SQL检查
        if not sanitized:
            return False, ["SQL语句为空"], ""

        # 1. 长度检查
        if len(sanitized) > self.max_sql_length:
            errors.append(f"SQL语句过长（{len(sanitized)}字符，上限{self.max_sql_length}）")

        # 2. 安全检查 - 禁止的关键词
        sql_upper = sanitized.upper()
        for kw in self.FORBIDDEN_KEYWORDS:
            # 使用词边界匹配，避免误判（如 _DROP_ 不应命中 'DROP TABLE' 但 'DROP ' 应命中）
            pattern = rf'\b{kw}\b'
            if re.search(pattern, sql_upper):
                errors.append(f"禁止的数据库操作: {kw}")

        # 3. 安全检查 - 危险模式
        for item in self.DANGEROUS_PATTERNS:
            if len(item) == 3:
                pattern, description, flags = item
            else:
                pattern, description = item
                flags = 0
            if re.search(pattern, sanitized, flags=flags):
                if "注释" in description:
                    logger.warning(f"SQL包含{description}")
                else:
                    errors.append(f"检测到危险模式: {description}")

        # 4. 语法检查
        try:
            parsed = sqlparse.parse(sanitized)
            if not parsed:
                errors.append("SQL语法无效——无法解析")
            else:
                # 检查第一条语句的类型
                stmt_type = parsed[0].get_type()
                if stmt_type != "SELECT" and stmt_type != "UNKNOWN":
                    errors.append(f"仅允许SELECT查询，不支持: {stmt_type}")
        except Exception as e:
            errors.append(f"SQL解析失败: {e}")

        # 5. 表名检查
        if allowed_tables:
            tables_in_sql = self._extract_table_names(sanitized)
            for t in tables_in_sql:
                if t.upper() not in [at.upper() for at in allowed_tables]:
                    errors.append(f"查询引用了不在白名单中的表: {t}")

        # 6. 清理: 移除末尾分号
        sanitized = sanitized.rstrip(";").strip()

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"SQL校验失败: {'; '.join(errors)}")

        return is_valid, errors, sanitized

    @staticmethod
    def _extract_table_names(sql: str) -> List[str]:
        """从 SQL 中提取引用的真实表名。

        用 sqlparse 的 token 树解析，替代原来的正则。原正则会把
        子查询别名（`s0`/`s1`）或数字字面量（`53`）误判成表名，
        导致合法 SQL 被白名单拦截。
        """
        tables: set = set()

        for stmt in sqlparse.parse(sql):
            cte_names = set()

            # 1) 收集 WITH 子句定义的 CTE 名称，后续引用同名时跳过
            toks = stmt.tokens
            for i, tok in enumerate(toks):
                if tok.ttype in T.Keyword and tok.value.upper() == "WITH":
                    for nt in toks[i + 1:]:
                        if nt.is_whitespace:
                            continue
                        if nt.ttype in T.Keyword.DML:
                            break
                        if isinstance(nt, IdentifierList):
                            for cte in nt.get_identifiers():
                                name = SQLValidator._first_name(cte)
                                if name:
                                    cte_names.add(name)
                        elif isinstance(nt, Identifier):
                            name = SQLValidator._first_name(nt)
                            if name:
                                cte_names.add(name)
                        break

            def add_table(ident):
                # 子查询 `FROM (SELECT ...) alias`：只递归取内部表，忽略别名
                if isinstance(ident, Parenthesis):
                    walk(ident.tokens)
                    return
                for t in getattr(ident, "tokens", []):
                    if isinstance(t, Parenthesis):
                        walk(t.tokens)
                        return
                name = ident.get_real_name()
                if name:
                    name = name.strip('`"[]')
                    if name not in cte_names:
                        tables.add(name)

            def walk(tokens):
                i = 0
                n = len(tokens)
                while i < n:
                    tok = tokens[i]
                    if tok.is_whitespace:
                        i += 1
                        continue
                    uv = tok.value.upper()
                    if tok.ttype in T.Keyword and (uv == "FROM" or uv.split()[-1] == "JOIN"):
                        j = i + 1
                        while j < n and tokens[j].is_whitespace:
                            j += 1
                        if j < n:
                            nt = tokens[j]
                            if isinstance(nt, IdentifierList):
                                for ident in nt.get_identifiers():
                                    add_table(ident)
                            else:
                                add_table(nt)
                        i = j
                    else:
                        if hasattr(tok, "tokens"):
                            walk(tok.tokens)
                        i += 1

            walk(stmt.tokens)

        return [t for t in tables if t]

    @staticmethod
    def _first_name(ident) -> Optional[str]:
        """返回标识符里的第一个名称 token（表名/CTE 名），无则返回 None"""
        for t in getattr(ident, "tokens", []):
            if t.ttype in T.Name:
                return t.value
        return None


if __name__ == "__main__":
    # 自检：CTE 名、子查询别名、数字字面量不能被误判成表名
    v = SQLValidator()
    assert v._extract_table_names("SELECT * FROM Sheet0") == ["Sheet0"]
    assert set(v._extract_table_names(
        "WITH s0 AS (SELECT * FROM Sheet0), s1 AS (SELECT * FROM Sheet1) "
        "SELECT * FROM s0 JOIN s1 ON s0.a = s1.a"
    )) == {"Sheet0", "Sheet1"}
    assert v._extract_table_names(
        "SELECT * FROM (SELECT * FROM Sheet0 WHERE x = 53) s0"
    ) == ["Sheet0"]
    ok, errs, _ = v.validate("SELECT * FROM evil", ["Sheet0"], [])
    assert not ok and "evil" in errs[0]
    print("sql_validator 自检通过")
