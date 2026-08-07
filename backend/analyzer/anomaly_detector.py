"""
异常检测器 (M4-3 / Phase 3)
对查询结果自动检测异常/离群数据点，支持三种方法:
- IQR (四分位距法): 零依赖，适合单变量分布检测
- Z-Score (标准差法): 零依赖，适合近似正态分布
- Isolation Forest (孤立森林): 基于 sklearn，适合多维异常检测
"""
import logging
from typing import List, Optional
import pandas as pd
import numpy as np

from backend.shared.schemas import (
    QueryResult, AnomalyPoint, AnomalyResult, ChartConfig, ChartType,
)

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    异常检测器

    自动选择检测方法:
    1. 单数值列 + 行数≥10 → IQR (快速、直观)
    2. 单数值列 + 行数≥30 → Z-Score (适合正态分布)
    3. 多数值列 + 行数≥50 → Isolation Forest (多维检测)

    用法:
        detector = AnomalyDetector()
        result = detector.detect(query_result)
        # result.anomalies → 异常点列表
        # result.chart       → 异常标注散点图
    """

    # 检测方法阈值
    MIN_ROWS_IQR = 10
    MIN_ROWS_ZSCORE = 30
    MIN_ROWS_IFOREST = 50
    ZSCORE_THRESHOLD = 3.0          # |z| > 3 判定为异常
    IQR_MULTIPLIER = 1.5            # 标准IQR系数
    IFOREST_CONTAMINATION = 0.05    # 预期异常比例 5%

    def detect(
        self,
        result: QueryResult,
        method: str = "auto",
    ) -> AnomalyResult:
        """
        执行异常检测

        Args:
            result: 查询执行结果
            method: 检测方法 ("auto" | "iqr" | "zscore" | "isolation_forest")

        Returns:
            AnomalyResult
        """
        if not result.success or not result.data or result.row_count == 0:
            return AnomalyResult(
                total_rows=result.row_count,
                method="none",
            )

        df = pd.DataFrame(result.data, columns=result.columns)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            return AnomalyResult(
                total_rows=len(df),
                method="none (无数值列)",
            )

        n_rows = len(df)

        # 自动选择方法
        if method == "auto":
            if len(num_cols) >= 2 and n_rows >= self.MIN_ROWS_IFOREST:
                method = "isolation_forest"
            elif n_rows >= self.MIN_ROWS_ZSCORE:
                method = "zscore"
            elif n_rows >= self.MIN_ROWS_IQR:
                method = "iqr"
            else:
                return AnomalyResult(
                    total_rows=n_rows,
                    method=f"insufficient_data (需要≥{self.MIN_ROWS_IQR}行)",
                )

        logger.info(
            f"异常检测 | 方法: {method} | 行数: {n_rows} | 数值列: {num_cols}"
        )

        if method == "iqr":
            anomalies = self._detect_iqr(df, num_cols)
        elif method == "zscore":
            anomalies = self._detect_zscore(df, num_cols)
        elif method == "isolation_forest":
            anomalies = self._detect_isolation_forest(df, num_cols)
        else:
            anomalies = []

        # 构建异常标注散点图
        chart = self._build_anomaly_chart(df, num_cols, anomalies) if anomalies else None

        return AnomalyResult(
            anomalies=anomalies,
            total_rows=n_rows,
            anomaly_count=len(anomalies),
            anomaly_rate=round(len(anomalies) / n_rows, 4) if n_rows > 0 else 0.0,
            method=method,
            chart=chart,
        )

    # ---- 检测方法 ----

    def _detect_iqr(
        self, df: pd.DataFrame, num_cols: List[str]
    ) -> List[AnomalyPoint]:
        """IQR 四分位距法"""
        anomalies = []
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < self.MIN_ROWS_IQR:
                continue

            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.IQR_MULTIPLIER * IQR
            upper = Q3 + self.IQR_MULTIPLIER * IQR

            mask = (df[col] < lower) | (df[col] > upper)
            for idx in df.index[mask]:
                val = float(df.loc[idx, col])
                if val < lower:
                    reason = f"{col}={val:.2f} 低于下界{lower:.2f}（IQR法）"
                else:
                    reason = f"{col}={val:.2f} 超过上界{upper:.2f}（IQR法）"

                # 计算近似的 Z-Score 用于排序
                mean, std = series.mean(), series.std()
                z = abs((val - mean) / std) if std > 0 else 0

                severity = self._judge_severity(z)
                anomalies.append(AnomalyPoint(
                    row_index=int(idx),
                    column=col,
                    value=val,
                    z_score=round(float(z), 2),
                    severity=severity,
                    reason=reason,
                ))

        return sorted(anomalies, key=lambda a: abs(a.z_score), reverse=True)

    def _detect_zscore(
        self, df: pd.DataFrame, num_cols: List[str]
    ) -> List[AnomalyPoint]:
        """Z-Score 标准差法"""
        anomalies = []
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < self.MIN_ROWS_ZSCORE:
                continue

            mean = series.mean()
            std = series.std()
            if std == 0:
                continue  # 常数列，无异常

            z_scores = (df[col] - mean) / std
            mask = abs(z_scores) > self.ZSCORE_THRESHOLD

            for idx in df.index[mask]:
                val = float(df.loc[idx, col])
                z = float(z_scores[idx])
                severity = self._judge_severity(abs(z))
                direction = "偏高" if z > 0 else "偏低"
                anomalies.append(AnomalyPoint(
                    row_index=int(idx),
                    column=col,
                    value=val,
                    z_score=round(abs(z), 2),
                    severity=severity,
                    reason=f"{col}={val:.2f} {direction}（Z={z:.2f}，阈值={self.ZSCORE_THRESHOLD}）",
                ))

        return sorted(anomalies, key=lambda a: abs(a.z_score), reverse=True)

    def _detect_isolation_forest(
        self, df: pd.DataFrame, num_cols: List[str]
    ) -> List[AnomalyPoint]:
        """Isolation Forest 多维异常检测"""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("sklearn 未安装，回退到 Z-Score 方法")
            return self._detect_zscore(df, num_cols)

        # 填充缺失值
        X = df[num_cols].fillna(df[num_cols].median()).values

        model = IsolationForest(
            contamination=self.IFOREST_CONTAMINATION,
            random_state=42,
            n_estimators=100,
        )
        preds = model.fit_predict(X)  # 1=正常, -1=异常
        scores = model.decision_function(X)  # 越高越正常

        anomalies = []
        for i, (pred, score) in enumerate(zip(preds, scores)):
            if pred == -1:
                # 找到该行偏离最大的列
                row = X[i]
                col_means = np.nanmean(X, axis=0)
                col_stds = np.nanstd(X, axis=0)
                col_stds[col_stds == 0] = 1.0
                z_scores = abs((row - col_means) / col_stds)
                worst_col_idx = int(np.argmax(z_scores))
                worst_col = num_cols[worst_col_idx]
                worst_z = float(z_scores[worst_col_idx])

                severity = self._judge_severity(worst_z)
                anomalies.append(AnomalyPoint(
                    row_index=i,
                    column=worst_col,
                    value=float(df.iloc[i][worst_col]),
                    z_score=round(worst_z, 2),
                    severity=severity,
                    reason=(
                        f"多维异常 | 最突出: {worst_col}={df.iloc[i][worst_col]:.2f} "
                        f"（Z={worst_z:.2f}，IsolationForest异常分={score:.3f}）"
                    ),
                ))

        return sorted(anomalies, key=lambda a: abs(a.z_score), reverse=True)

    # ---- 图表构建 ----

    def _build_anomaly_chart(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        anomalies: List[AnomalyPoint],
    ) -> ChartConfig:
        """构建异常标注散点图 ECharts option"""
        import json

        x_col = num_cols[0]
        y_col = num_cols[1] if len(num_cols) >= 2 else num_cols[0]

        # 正常点
        anomalous_indices = set(a.row_index for a in anomalies)
        normal_x = [float(df.iloc[i][x_col]) for i in range(len(df)) if i not in anomalous_indices and pd.notna(df.iloc[i][x_col]) and pd.notna(df.iloc[i][y_col])]
        normal_y = [float(df.iloc[i][y_col]) for i in range(len(df)) if i not in anomalous_indices and pd.notna(df.iloc[i][x_col]) and pd.notna(df.iloc[i][y_col])]

        # 异常点
        anomaly_indices = sorted(anomalous_indices)
        anomaly_x = [float(df.iloc[i][x_col]) for i in anomaly_indices if pd.notna(df.iloc[i][x_col]) and pd.notna(df.iloc[i][y_col])]
        anomaly_y = [float(df.iloc[i][y_col]) for i in anomaly_indices if pd.notna(df.iloc[i][x_col]) and pd.notna(df.iloc[i][y_col])]

        option = {
            "title": {
                "text": f"异常检测 ({len(anomalies)}个异常点)",
                "left": "center",
                "textStyle": {"fontSize": 16},
            },
            "tooltip": {
                "trigger": "item",
                "formatter": f"{x_col}: {{c[0]}}<br/>{y_col}: {{c[1]}}",
            },
            "legend": {"bottom": 0, "data": ["正常", "异常"]},
            "xAxis": {"type": "value", "name": x_col},
            "yAxis": {"type": "value", "name": y_col},
            "color": ["#5470c6", "#ee6666"],
            "series": [
                {
                    "name": "正常",
                    "type": "scatter",
                    "data": [[x, y] for x, y in zip(normal_x, normal_y)],
                    "symbolSize": 8,
                },
                {
                    "name": "异常",
                    "type": "scatter",
                    "data": [[x, y] for x, y in zip(anomaly_x, anomaly_y)],
                    "symbolSize": 14,
                    "emphasis": {"scaleSize": 18},
                },
            ],
            "toolbox": {
                "feature": {
                    "saveAsImage": {"title": "保存为图片"},
                },
            },
        }

        return ChartConfig(
            chart_type=ChartType.SCATTER,
            title=f"异常检测 ({len(anomalies)}个异常点)",
            x_axis=x_col,
            y_axis=y_col,
            echarts_option=option,
        )

    # ---- 工具 ----

    @staticmethod
    def _judge_severity(z_abs: float) -> str:
        """根据 Z-Score 绝对值判定严重程度"""
        if z_abs >= 5.0:
            return "high"
        elif z_abs >= 3.5:
            return "medium"
        else:
            return "low"
