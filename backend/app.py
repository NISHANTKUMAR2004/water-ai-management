import sys
import os
from flask import Flask, render_template, request

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from ml.predict_cluster import predict_cluster

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)


@app.route("/")
def home():
    # 🔥 THIS LOADS THE UI
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_input = {
        "annual": request.form["annual"],
        "groundwater": request.form["groundwater"],
        "rice": request.form["rice"],
        "wheat": request.form["wheat"],
        "windspeed": request.form["windspeed"],
        "sugarcane": request.form["sugarcane"],
        "dugwell_count": request.form["dugwell_count"],
    }

    cluster = predict_cluster(user_input)

    if cluster == 0:
        status = "⚠️ Water-Stressed Region"
        suggestion = (
            "High dependency on groundwater detected. "
            "Adopt rainwater harvesting, crop diversification, "
            "and reduce water-intensive crops."
        )
    else:
        status = "✅ Water-Sustainable Region"
        suggestion = (
            "Balanced rainfall and irrigation observed. "
            "Continue efficient irrigation and sustainable practices."
        )

    return render_template(
        "result.html",
        cluster=cluster,
        status=status,
        suggestion=suggestion,
        inputs=user_input
    )


# Optional health check (useful for Render)
@app.route("/api/status")
def api_status():
    return {"status": "Water AI backend running"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

app.run(host="0.0.0.0", port=10000)
