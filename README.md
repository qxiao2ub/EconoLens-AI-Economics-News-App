# EconoLens AI

**Economics News Simplifier and Personal Impact Storyteller**

**Author:** Rishabh Shah  
**Advisor:** Dr. Qingyang Xiao  
**Student level:** Senior Student

EconoLens AI converts a long, terminology-heavy economics article into a layered, plain-language explanation. It preserves source evidence, explains economics terminology, tells the economic story, and shows possible pathways through which the article could affect everyday life.

## What the app includes

- Five-level information hierarchy, from a 15-second takeaway to evidence and outlook
- TF-IDF and maximal marginal relevance for extractive evidence selection
- Three-hidden-layer MLP neural network for economic-topic classification
- Economics terminology detection and plain-language definitions
- Grounded story generation and reader-profile personal-impact analysis
- Transparent n-gram next-word forecasting demonstration
- Optional FLAN-T5 neural rewrite with number-grounding checks
- Epsilon-greedy multi-armed bandit for explanation-format feedback
- Readability diagnostics, quality warnings, Markdown export, and JSON export

## Repository structure

```text
.
|-- app.py                         # Streamlit web application
|-- econolens_core.py              # Economics, ML, DNN, NLP, and feedback engine
|-- sample_article.txt             # Fictional demonstration article
|-- requirements.txt               # Default Streamlit Cloud dependencies
|-- requirements-neural.txt        # Optional FLAN-T5 dependencies
|-- LICENSE                        # MIT License
|-- AUTHORS.md                     # Author and advisor credits
|-- notebooks/
|   `-- Rishabh_Shah_EconoLens_AI_Colab.ipynb
|-- tests/
|   `-- test_core.py
`-- .streamlit/
    `-- config.toml
```

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat

# macOS or Linux
source .venv/bin/activate
```

Install the default dependencies and run the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The default installation supports the complete deterministic workflow. No API key is required.

### Optional neural rewrite

The FLAN-T5 feature is intentionally separated because PyTorch and Transformers make cloud builds larger. Install it only when needed:

```bash
pip install -r requirements-neural.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Extract this ZIP file.
2. Create a new GitHub repository.
3. Upload all extracted repository files and folders to the repository root.
4. In Streamlit Community Cloud, create an app from the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy the app and review the build log.
7. Test the included synthetic article before trying pasted text, uploads, or public URLs.

For the lightest and most reliable deployment, keep `requirements.txt` as the dependency file. The optional neural rewrite can be enabled later by adding the packages from `requirements-neural.txt` to the deployed environment.

## Testing

Run the deterministic smoke tests with:

```bash
python -m unittest discover -s tests -v
```

## Responsible-use notes

- This is an educational prototype, not financial, investment, legal, or tax advice.
- Verify important statements against the original article and current official data.
- Personal-impact cards describe possible pathways, not guaranteed outcomes.
- Public URL extraction does not bypass paywalls or access controls.
- The feedback learner changes presentation preferences only; it does not change facts.
- Do not collect sensitive user feedback without adding authentication, a privacy notice, and secure persistent storage.

## Credits

- **Author:** Rishabh Shah
- **Advisor:** Dr. Qingyang Xiao

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
