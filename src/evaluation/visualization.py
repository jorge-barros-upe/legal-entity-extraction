"""
Visualization utilities for experiment results.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Visualize experiment results."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir

    def plot_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "f1",
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot comparison bar chart of models.

        Args:
            results: {model_name: {metric: value}}
            metric: Which metric to plot
            title: Chart title
            save_path: Where to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            models = list(results.keys())
            values = [results[m].get(metric, 0) for m in models]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(models, values, color='steelblue')

            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_ylim(0, 1.0)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom')

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot to {save_path}")

            return fig

        except ImportError:
            logger.warning("matplotlib not installed. Skipping visualization.")
            return None

    def plot_metrics_radar(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ):
        """Plot radar chart comparing multiple metrics across models."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if metrics is None:
                metrics = ["precision", "recall", "f1", "coverage", "consistency"]

            models = list(results.keys())
            num_metrics = len(metrics)

            angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

            for model in models:
                values = [results[model].get(m, 0) for m in metrics]
                values += values[:1]  # Complete the circle
                ax.plot(angles, values, 'o-', linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title("Model Performance Comparison")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

        except ImportError:
            logger.warning("matplotlib not installed.")
            return None

    def plot_by_entity_type(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = "f1",
        save_path: Optional[str] = None
    ):
        """Plot performance by entity type."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            models = list(results.keys())
            first_model = models[0]
            entity_types = list(results[first_model].keys())

            x = np.arange(len(entity_types))
            width = 0.8 / len(models)

            fig, ax = plt.subplots(figsize=(14, 6))

            for i, model in enumerate(models):
                values = [results[model].get(et, {}).get(metric, 0) for et in entity_types]
                ax.bar(x + i * width, values, width, label=model)

            ax.set_ylabel(metric.upper())
            ax.set_title(f'Performance by Entity Type ({metric})')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(entity_types, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.0)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

        except ImportError:
            return None

    def generate_latex_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        caption: str = "Model Comparison Results"
    ) -> str:
        """Generate LaTeX table from results."""
        if metrics is None:
            metrics = ["precision", "recall", "f1"]

        models = list(results.keys())

        # Build table
        header = " & ".join(["Model"] + [m.capitalize() for m in metrics])
        rows = []

        for model in models:
            values = [model] + [f"{results[model].get(m, 0):.4f}" for m in metrics]
            rows.append(" & ".join(values))

        rows_str = ' \\\\\n'.join(rows)
        cols_spec = 'l' + 'c' * len(metrics)
        table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\begin{{tabular}}{{{cols_spec}}}
\\toprule
{header} \\\\
\\midrule
{rows_str} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        return table

    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """Generate comprehensive HTML report."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Entity Extraction Experiment Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { font-weight: bold; color: #2196F3; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
    </style>
</head>
<body>
    <h1>Entity Extraction Experiment Results</h1>
"""

        # Add results tables
        for model_name, model_results in results.items():
            html += f"<h2>{model_name}</h2>"
            html += "<table><tr><th>Metric</th><th>Value</th></tr>"

            if isinstance(model_results, dict):
                for metric, value in model_results.items():
                    if isinstance(value, float):
                        html += f"<tr><td>{metric}</td><td class='metric'>{value:.4f}</td></tr>"
                    else:
                        html += f"<tr><td>{metric}</td><td>{value}</td></tr>"

            html += "</table>"

        html += """
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"Generated report: {output_path}")
