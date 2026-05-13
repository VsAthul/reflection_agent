from state import FAQAgentState, QuestionResult, ValidationResult


def save_result_node(state: FAQAgentState) -> FAQAgentState:
    """
    LangGraph node: Saves the result for the current question (valid or exhausted retries)
    into state["results"] and advances current_index to the next question.
    """
    validation: ValidationResult | None = state.get("validation")
    retry_count  = state["retry_count"]
    answer_text  = state["generated_answer"]

    # Determine final status
    if validation and validation.is_valid:
        status        = "answered"
        final_answer  = answer_text
        remarks       = validation.remarks
        checked       = True
    else:
        # Exhausted retries without a valid answer
        status        = "no_answer"
        final_answer  = "-"
        remarks       = (
            validation.remarks
            if validation
            else "Max retries reached without a valid answer."
        )
        checked       = validation.checked_answer if validation else False

    result: QuestionResult = {
        "faqQuestion":    state["current_faq_id"],
        "Question":       state["current_question"],
        "iterationCount": retry_count,
        "status":         status,
        "Answer":         final_answer,
        "checkedAnswer":  checked,
        "remarks":        remarks,
    }

    print(
        f"[SaveResult] Q{state['current_faq_id']} → status={status} "
        f"| iterations={retry_count}"
    )

    updated_results = state["results"] + [result]
    return {
        **state,
        "results":       updated_results,
        "current_index": state["current_index"] + 1,
    }