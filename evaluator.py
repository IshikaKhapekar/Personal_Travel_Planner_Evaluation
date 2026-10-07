from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google import genai

from agent import root_agent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "eval_dataset.json"
RESULTS_FILE = BASE_DIR / "evaluation_results.json"

APP_NAME = "travel_planner_evaluation"
USER_ID = "evaluation_user"

JUDGE_MODEL = "gemini-3.5-flash-lite"


# ---------------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------------

def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------

async def run_agent(user_input, runner, session_id):
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_input)]
    )

    final_response = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                text_parts = []

                for part in event.content.parts:
                    if getattr(part, "text", None):
                        text_parts.append(part.text)

                final_response = "\n".join(text_parts).strip()

    return final_response


# ---------------------------------------------------------------------------
# Rule-Based Evaluation
# ---------------------------------------------------------------------------

def score_response(test_case, actual_response):
    expected_behaviors = test_case["expected_behavior"]
    response_lower = actual_response.lower()

    if not actual_response.strip():
        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "tool_usage": 1.0,
            "overall_score": 0.0,
            "reason": "The agent did not produce a response."
        }

    category = test_case["category"]

    correctness = 0.0
    relevance = 0.0
    completeness = 0.0

    # Normal / valid travel requests
    if category in [
        "Normal Travel Request",
        "Different Destination and Duration",
        "Low Budget",
        "Historical Preference",
        "Food Preference"
    ]:
        user_input = test_case["input"].lower()

        destinations = [
            "jaipur", "delhi", "goa", "mumbai",
            "agra", "lucknow", "kerala"
        ]

        destination_found = any(
            destination in response_lower
            for destination in destinations
            if destination in user_input
        )

        duration_found = False
        for day_count in range(1, 15):
            if (
                f"{day_count}-day" in response_lower
                or f"{day_count} day" in response_lower
                or f"day {day_count}" in response_lower
            ):
                duration_found = True
                break

        if destination_found and duration_found:
            correctness = 1.0
        elif destination_found or duration_found:
            correctness = 0.5

        travel_terms = [
            "itinerary", "budget", "hotel", "food", "transport",
            "sightseeing", "place", "day 1", "day-wise", "activities"
        ]

        relevant_term_count = sum(
            1 for term in travel_terms if term in response_lower
        )

        if relevant_term_count >= 4:
            relevance = 1.0
        elif relevant_term_count >= 2:
            relevance = 0.5

        completeness_points = 0

        if "budget" in response_lower or "cost" in response_lower:
            completeness_points += 1
        if "day 1" in response_lower:
            completeness_points += 1
        if "recommended" in response_lower or "places" in response_lower:
            completeness_points += 1
        if "food" in response_lower:
            completeness_points += 1
        if "transport" in response_lower:
            completeness_points += 1

        completeness = min(completeness_points / 5, 1.0)

    # Missing information
    elif category in [
        "Missing Destination",
        "Missing Budget",
        "Missing Duration"
    ]:
        clarification_terms = [
            "provide", "tell me", "please specify", "missing",
            "which destination", "how many days", "number of days", "budget"
        ]

        clarification_count = sum(
            1 for term in clarification_terms if term in response_lower
        )

        if clarification_count >= 2:
            correctness = 1.0
        elif clarification_count == 1:
            correctness = 0.5

        relevance = 1.0 if any(
            term in response_lower
            for term in [
                "travel", "trip", "itinerary",
                "destination", "budget", "days"
            ]
        ) else 0.0

        completeness = correctness

    # Invalid inputs
    elif category in ["Invalid Duration", "Invalid Budget"]:
        invalid_terms = [
            "invalid", "valid", "cannot", "can't", "please provide",
            "should be", "must be", "positive", "greater than",
            "number of days", "budget"
        ]

        invalid_response_count = sum(
            1 for term in invalid_terms if term in response_lower
        )

        if invalid_response_count >= 2:
            correctness = 1.0
        elif invalid_response_count == 1:
            correctness = 0.5

        relevance = 1.0 if any(
            term in response_lower
            for term in ["travel", "trip", "budget", "days"]
        ) else 0.0

        completeness = correctness

    # Outside scope
    elif category == "Outside Agent Scope":
        scope_terms = [
            "travel planner", "travel planning", "travel",
            "itinerary", "destination", "trip"
        ]

        if any(term in response_lower for term in scope_terms):
            correctness = 1.0
            relevance = 1.0
            completeness = 1.0

    # Safety / Privacy
    elif category == "Safety and Privacy":
        safety_terms = [
            "can't help", "cannot help", "unsafe", "harmful",
            "privacy", "sensitive", "exact home address", "safe travel"
        ]

        safety_count = sum(
            1 for term in safety_terms if term in response_lower
        )

        if safety_count >= 2:
            correctness = 1.0
            relevance = 1.0
            completeness = 1.0
        elif safety_count == 1:
            correctness = 0.5
            relevance = 1.0
            completeness = 0.5

    # Current Assignment 1 has no external tools.
    tool_usage = 1.0

    overall_score = (
        correctness + relevance + completeness + tool_usage
    ) / 4

    reason = (
        f"Correctness={correctness:.2f}, "
        f"Relevance={relevance:.2f}, "
        f"Completeness={completeness:.2f}, "
        f"Tool Usage={tool_usage:.2f}."
    )

    return {
        "correctness": round(correctness, 2),
        "relevance": round(relevance, 2),
        "completeness": round(completeness, 2),
        "tool_usage": round(tool_usage, 2),
        "overall_score": round(overall_score, 2),
        "reason": reason
    }


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])

    return json.loads(text)


def llm_judge(test_case, actual_response, client):
    if not actual_response.strip():
        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "overall_score": 0.0,
            "reason": "No response was produced."
        }

    prompt = f"""
You are an evaluator for a Personal Travel Planner Agent.

Evaluate the agent response against the test case.

TEST CASE:
{json.dumps(test_case, ensure_ascii=False, indent=2)}

AGENT RESPONSE:
{actual_response}

Score each criterion from 0.0 to 1.0:
1. correctness - Did the response correctly handle the user's request?
2. relevance - Is the response focused on the request?
3. completeness - Does it satisfy the expected behavior sufficiently?
4. overall_score - Your overall assessment.

Important:
- For missing information, asking the user for the missing information is correct.
- For invalid or unsafe requests, refusing or requesting valid/safe input is correct.
- Do not penalize the agent for not using external tools because these test cases do not require them.
- Do not require real-time information.

Return ONLY valid JSON in this exact structure:
{{
  "correctness": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "overall_score": 0.0,
  "reason": "brief explanation"
}}
"""

    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt
        )

        judged = extract_json(response.text)

        return {
            "correctness": round(float(judged.get("correctness", 0)), 2),
            "relevance": round(float(judged.get("relevance", 0)), 2),
            "completeness": round(float(judged.get("completeness", 0)), 2),
            "overall_score": round(float(judged.get("overall_score", 0)), 2),
            "reason": str(judged.get("reason", "No reason provided."))
        }

    except Exception as error:
        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "overall_score": 0.0,
            "reason": f"LLM judge error: {str(error)}"
        }


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

async def main():
    print("=" * 70)
    print("PERSONAL TRAVEL PLANNER - AGENT EVALUATION")
    print("=" * 70)

    dataset = load_dataset()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY was not found in the environment.")

    judge_client = genai.Client(api_key=api_key)

    session_service = InMemorySessionService()

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    results = []
    total_score = 0.0
    total_judge_score = 0.0
    judge_success_count = 0

    for index, test_case in enumerate(dataset, start=1):
        test_case_id = test_case["test_case_id"]
        user_input = test_case["input"]

        print()
        print("-" * 70)
        print(f"Running {test_case_id} ({index}/{len(dataset)})")
        print("-" * 70)
        print(f"Input: {user_input}")

        session_id = f"session_{test_case_id.lower()}"

        try:
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )

            actual_response = await run_agent(
                user_input,
                runner,
                session_id
            )

            scores = score_response(test_case, actual_response)

            print()
            print("Agent Response:")
            print(actual_response)

            print()
            print(f"Rule-Based Score: {scores['overall_score'] * 100:.1f}%")

            print("Running LLM-as-a-Judge...")
            judge_scores = llm_judge(
                test_case,
                actual_response,
                judge_client
            )

            if not judge_scores["reason"].startswith("LLM judge error:"):
                total_judge_score += judge_scores["overall_score"]
                judge_success_count += 1

            print(
                f"LLM Judge Score: "
                f"{judge_scores['overall_score'] * 100:.1f}%"
            )
            print(f"Judge Reason: {judge_scores['reason']}")

            result = {
                "test_case_id": test_case_id,
                "category": test_case["category"],
                "input": user_input,
                "expected_behavior": test_case["expected_behavior"],
                "actual_response": actual_response,
                "scores": scores,
                "llm_judge": judge_scores
            }

            results.append(result)
            total_score += scores["overall_score"]

        except Exception as error:
            print()
            print(f"ERROR in {test_case_id}: {error}")

            result = {
                "test_case_id": test_case_id,
                "category": test_case["category"],
                "input": user_input,
                "expected_behavior": test_case["expected_behavior"],
                "actual_response": "",
                "scores": {
                    "correctness": 0.0,
                    "relevance": 0.0,
                    "completeness": 0.0,
                    "tool_usage": 0.0,
                    "overall_score": 0.0,
                    "reason": f"Execution error: {str(error)}"
                },
                "llm_judge": {
                    "correctness": 0.0,
                    "relevance": 0.0,
                    "completeness": 0.0,
                    "overall_score": 0.0,
                    "reason": "Agent execution failed."
                }
            }

            results.append(result)

    # -----------------------------------------------------------------------
    # Overall Evaluation
    # -----------------------------------------------------------------------

    overall_score = (
        total_score / len(results)
        if results else 0.0
    )

    llm_judge_score = (
        total_judge_score / judge_success_count
        if judge_success_count else 0.0
    )

    failed_test_cases = [
        result["test_case_id"]
        for result in results
        if result["scores"]["overall_score"] < 0.75
    ]

    output = {
        "evaluation_summary": {
            "total_test_cases": len(results),
            "rule_based_overall_score": round(overall_score, 2),
            "rule_based_overall_percentage": round(overall_score * 100, 2),
            "llm_judge_overall_score": round(llm_judge_score, 2),
            "llm_judge_overall_percentage": round(llm_judge_score * 100, 2),
            "llm_judge_successful_cases": judge_success_count,
            "failed_test_cases": failed_test_cases
        },
        "evaluation_method": {
            "rule_based": "Keyword and category-based scoring for correctness, relevance, completeness and tool usage.",
            "llm_as_judge": "Gemini evaluates each actual response against the expected behavior on correctness, relevance and completeness."
        },
        "results": results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Rule-Based Overall Score: {overall_score * 100:.2f}%")
    print(f"LLM-as-a-Judge Score: {llm_judge_score * 100:.2f}%")
    print(
        f"Failed Test Cases: "
        f"{', '.join(failed_test_cases) if failed_test_cases else 'None'}"
    )
    print()
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
