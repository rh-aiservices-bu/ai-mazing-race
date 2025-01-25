import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from trustyai.utils._visualisation import (
    DEFAULT_STYLE as ds,
    DEFAULT_RC_PARAMS as drcp,
)

def _matplotlib_plot(explanations, output_name=None, block=True, call_show=True) -> None:
    with mpl.rc_context(drcp):
        shap_values = [pfi.getScore() for pfi in explanations.saliency_map()[output_name].getPerFeatureImportance()[:-1]]
        feature_names = [str(pfi.getFeature().getName()) for pfi in explanations.saliency_map()[output_name].getPerFeatureImportance()[:-1]]
        fnull = explanations.get_fnull()[output_name]
        prediction = fnull + sum(shap_values)

        if call_show:
            plt.figure()

        pos = fnull
        for j, shap_value in enumerate(shap_values):
            color = ds["negative_primary_colour"] if shap_value < 0 else ds["positive_primary_colour"]
            width = 0.9
            if j > 0:
                plt.plot([j - 0.5, j + width / 2 * 0.99], [pos, pos], color=color)
            plt.bar(j, height=shap_value, bottom=pos, color=color, width=width)
            pos += shap_values[j]

            if j != len(shap_values) - 1:
                plt.plot([j - width / 2 * 0.99, j + 0.5], [pos, pos], color=color)

        plt.axhline(fnull, color="#444444", linestyle="--", zorder=0, label="Background Value")
        plt.axhline(prediction, color="#444444", zorder=0, label="Prediction")
        plt.legend()

        ticksize = np.diff(plt.gca().get_yticks())[0]
        plt.ylim(plt.gca().get_ylim()[0] - ticksize / 2, plt.gca().get_ylim()[1] + ticksize / 2)
        
        plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45, ha='right')  # Rotate labels and adjust alignment
        plt.tick_params(axis='x', labelsize=8)  # Adjust font size
        
        plt.ylabel(explanations.saliency_map()[output_name].getOutput().getName())
        plt.xlabel("Feature SHAP Value")
        plt.title(f"SHAP: Feature Contributions to {output_name}")
        
        if call_show:
            plt.show(block=block)
