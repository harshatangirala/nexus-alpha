"""Registry of knowledge-base documents. Exact filenames were checked against the Downloads directory listing
(the Kanu Jain lecture file name contains an underscore; 'Idenitfying' is the file's own spelling)."""
from __future__ import annotations

from .. import paths

_B = paths.BOOKS_DIR
_R = paths.RULEBOOK_DIR

BOOKS = {
    "complete_guide_to_trading": {"path": str(_B / "The-Complete-Guide-to-Trading.pdf"), "kind": "book",
                                  "supplied_for": "technical analysis RAG"},
    "tsam": {"path": str(_B / "TSaM.pdf"), "kind": "book", "supplied_for": "technical analysis RAG"},
    "nison_candlesticks": {"path": str(_B / "Japanese Candlestick Charting Techniques, 2nd Edition, Steve Nison_text.pdf"),
                           "kind": "book", "supplied_for": "technical analysis RAG"},
    "ta_workbook": {"path": str(_B / "TA_wrkbk.pdf"), "kind": "book", "supplied_for": "technical analysis RAG"},
    "identifying_chart_patterns": {"path": str(_B / "Idenitfying-Chart-Patterns.pdf"), "kind": "book",
                                   "supplied_for": "technical analysis RAG"},
    "fundamentals_of_investments_wk2": {
        "path": str(_B / "B.Com(Hons)_IIIyearVIsem_FundamentalsofInvestments_Week2_DrKanuJain.pdf"),
        "kind": "lecture_notes", "supplied_for": "technical analysis RAG"},
    "essentials_of_quant_finance": {"path": str(_B / "Essentials Of Quantitative Finance_1.pdf"), "kind": "book",
                                    "supplied_for": "related reading (user message)"},
    "dobelman_williams_qfa": {
        "path": str(_B / "Dobelman, John A._Williams, Edward E - Quantitative financial analytics_ the path to "
                         "investment profits-World Scientific (2018).pdf"),
        "kind": "book", "supplied_for": "related reading (user message)"},
}

RULEBOOKS = {
    "entry_rulebook_150": str(_R / "01_ENTRY_RULEBOOK_150.md"),
    "exit_rulebook_120": str(_R / "02_EXIT_RULEBOOK_120.md"),
    "mtf_decision_engine": str(_R / "03_MULTI_TIMEFRAME_DECISION_ENGINE.md"),
    "nn_training_spec": str(_R / "04_NEURAL_NETWORK_TRAINING_SPEC.md"),
    "data_contract_feature_dictionary": str(_R / "05_DATA_CONTRACT_AND_FEATURE_DICTIONARY.md"),
    "labeling_backtesting_validation": str(_R / "06_LABELING_BACKTESTING_AND_STATISTICAL_VALIDATION.md"),
    "colab_training_guide": str(_R / "11_COLAB_AND_TRAINING_EXECUTION_GUIDE.md"),
    "verification_and_handoff": str(_R / "12_VERIFICATION_AND_HANDOFF.md"),
    "three_timeframe_readme": str(_R / "README (1).md"),
}
