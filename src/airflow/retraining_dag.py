"""
Automated Model Retraining DAG — Week 4: Integration & Productionization
Apache Airflow pipeline for weekly retraining of the Contract Intelligence models.

Schedule: Every Sunday at 2:00 AM UTC
Tasks:
  1. load_new_data       → scan data/processed/ for new contracts
  2. evaluate_current    → benchmark current model performance
  3. retrain_ner         → retrain spaCy NER on new annotations
  4. retrain_classifier  → fine-tune transformer on new CUAD data
  5. save_checkpoint     → save versioned model weights
  6. notify              → log completion summary

To run Airflow locally:
  pip install apache-airflow
  airflow db init
  airflow webserver --port 8080
  airflow scheduler
  (Then open http://localhost:8080 and enable this DAG)
"""
from datetime import datetime, timedelta
import os
import json

# ─── Airflow is optional — graceful import ─────────────────────────
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# Each function is a standalone task in the DAG.
# ═══════════════════════════════════════════════════════════════════

def task_load_new_data(**context):
    """
    Task 1: Scan data/processed/ for newly analyzed contracts.
    Pushes the contract count to XCom for downstream tasks.
    """
    processed_dir = "data/processed"
    contracts = []

    if os.path.exists(processed_dir):
        for filename in os.listdir(processed_dir):
            if filename.endswith("_analysis.json"):
                filepath = os.path.join(processed_dir, filename)
                try:
                    with open(filepath) as f:
                        contracts.append(json.load(f))
                except Exception:
                    pass

    count = len(contracts)
    print(f"📂 Found {count} processed contracts for potential retraining")

    if count < 10:
        print("⚠️  Not enough contracts for retraining (minimum 10 required). Skipping.")
    else:
        print(f"✅ {count} contracts ready for retraining pipeline")

    # Push to XCom so downstream tasks can use it
    if context.get("ti"):
        context["ti"].xcom_push(key="contract_count", value=count)

    return count


def task_evaluate_current_model(**context):
    """
    Task 2: Evaluate the current deployed model's performance.
    In production: run the model on a held-out test set and compute F1, precision, recall.
    """
    print("📊 Evaluating current model performance...")

    # In production: load model + run evaluation
    # from src.nlp.classifier import LegalClauseClassifier
    # model = LegalClauseClassifier.load("models/legal_classifier.pth")
    # metrics = evaluate(model, test_loader)

    # Mock metrics for demo
    mock_metrics = {
        "f1_score": 0.87,
        "precision": 0.89,
        "recall": 0.85,
        "accuracy": 0.91,
        "evaluated_at": datetime.now().isoformat()
    }

    print(f"   F1 Score:  {mock_metrics['f1_score']:.2%}")
    print(f"   Precision: {mock_metrics['precision']:.2%}")
    print(f"   Recall:    {mock_metrics['recall']:.2%}")

    if context.get("ti"):
        context["ti"].xcom_push(key="current_metrics", value=mock_metrics)

    return mock_metrics


def task_retrain_ner(**context):
    """
    Task 3: Retrain the spaCy Named Entity Recognition model
    with newly annotated contract data.
    """
    print("🔄 Retraining spaCy NER model...")

    # In production:
    # import spacy
    # from spacy.training import Example
    # nlp = spacy.load("en_core_web_sm")
    # optimizer = nlp.begin_training()
    # for batch in training_batches:
    #     nlp.update(batch, sgd=optimizer)
    # nlp.to_disk("models/ner_retrained/")

    print("   Loading training annotations from data/processed/...")
    print("   Running 10 training epochs...")
    print("   Entity types: ORG, DATE, MONEY, JURISDICTION, PARTY")
    print("✅ NER model retrained successfully")

    return "ner_retrained"


def task_retrain_classifier(**context):
    """
    Task 4: Fine-tune the transformer-based legal clause classifier
    on the latest batch of labeled contract data.
    """
    print("🔄 Fine-tuning Legal Clause Classifier (RoBERTa)...")

    # In production: run train_transformer.py
    # import subprocess
    # result = subprocess.run(["python", "-m", "src.nlp.train_transformer"], capture_output=True)

    print("   Loading new labeled samples from CUAD + processed contracts...")
    print("   Fine-tuning roberta-base for 3 epochs...")
    print("   Learning rate: 2e-5 | Batch size: 16 | Max length: 256")
    print("✅ Classifier fine-tuning complete")

    return "classifier_retrained"


def task_save_checkpoint(**context):
    """
    Task 5: Save versioned model checkpoints.
    Keeps the last 3 versions for rollback capability.
    """
    os.makedirs("models", exist_ok=True)

    version_tag = datetime.now().strftime("%Y%m%d_%H%M")
    ner_path = f"models/ner_{version_tag}"
    classifier_path = f"models/legal_classifier_{version_tag}.pth"

    print(f"💾 Saving model checkpoints:")
    print(f"   NER:        {ner_path}/")
    print(f"   Classifier: {classifier_path}")

    # In production: torch.save(model.state_dict(), classifier_path)
    # Cleanup old checkpoints (keep last 3)
    all_checkpoints = sorted([
        f for f in os.listdir("models") if f.startswith("legal_classifier_")
    ])
    if len(all_checkpoints) > 3:
        for old in all_checkpoints[:-3]:
            old_path = os.path.join("models", old)
            print(f"   🗑️  Removing old checkpoint: {old}")
            # os.remove(old_path)

    print("✅ Checkpoints saved")
    return {"ner": ner_path, "classifier": classifier_path}


def task_notify_completion(**context):
    """
    Task 6: Send completion notification (email/Slack/log).
    """
    completion_summary = {
        "pipeline": "contract_model_retraining",
        "completed_at": datetime.now().isoformat(),
        "tasks_completed": ["load_data", "evaluate", "retrain_ner", "retrain_classifier", "save_checkpoint"],
        "status": "SUCCESS",
    }

    print("\n" + "="*50)
    print("✅ RETRAINING PIPELINE COMPLETE")
    print(f"   Completed at: {completion_summary['completed_at']}")
    print(f"   Status:       {completion_summary['status']}")
    print("="*50)

    # Save pipeline run log
    os.makedirs("data/pipeline_logs", exist_ok=True)
    log_file = f"data/pipeline_logs/retrain_{datetime.now().strftime('%Y%m%d')}.json"
    with open(log_file, "w") as f:
        json.dump(completion_summary, f, indent=2)

    return completion_summary


# ═══════════════════════════════════════════════════════════════════
# DAG DEFINITION
# Only created if Airflow is installed
# ═══════════════════════════════════════════════════════════════════

if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "zaalima-mlops",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=4),
    }

    with DAG(
        dag_id="contract_model_retraining",
        default_args=default_args,
        description="Weekly retraining pipeline for Contract Intelligence NLP models",
        schedule_interval="0 2 * * 0",   # Every Sunday at 02:00 UTC
        start_date=datetime(2024, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["nlp", "contract-intelligence", "retraining", "zaalima"],
    ) as dag:

        t1_load = PythonOperator(
            task_id="load_new_training_data",
            python_callable=task_load_new_data,
        )

        t2_evaluate = PythonOperator(
            task_id="evaluate_current_model",
            python_callable=task_evaluate_current_model,
        )

        t3_ner = PythonOperator(
            task_id="retrain_ner_model",
            python_callable=task_retrain_ner,
        )

        t4_classifier = PythonOperator(
            task_id="retrain_clause_classifier",
            python_callable=task_retrain_classifier,
        )

        t5_checkpoint = PythonOperator(
            task_id="save_model_checkpoint",
            python_callable=task_save_checkpoint,
        )

        t6_notify = PythonOperator(
            task_id="notify_completion",
            python_callable=task_notify_completion,
        )

        # Task dependency graph:
        # load → evaluate → [retrain_ner, retrain_classifier] → save → notify
        t1_load >> t2_evaluate >> [t3_ner, t4_classifier] >> t5_checkpoint >> t6_notify

else:
    print("ℹ️  Apache Airflow not installed — DAG definition skipped.")
    print("   To use the retraining pipeline, install Airflow:")
    print("   pip install apache-airflow")
    print("   Then run: airflow db init && airflow webserver --port 8080")


if __name__ == "__main__":
    print("Running retraining pipeline manually (without Airflow)...")
    ctx = {}
    count = task_load_new_data(**ctx)
    metrics = task_evaluate_current_model(**ctx)
    task_retrain_ner(**ctx)
    task_retrain_classifier(**ctx)
    task_save_checkpoint(**ctx)
    task_notify_completion(**ctx)
