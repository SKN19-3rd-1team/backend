import sys
from backend.main import run_mentor_stream, run_major_recommendation


def print_stream(generator):
    """
    Consumes the stream generator and prints content to console.
    Handles 'delta' (content), 'status' (tool usage), and 'error' messages.
    """
    sys.stdout.write("Agent: ")
    sys.stdout.flush()

    full_content = ""

    for mode, chunk in generator:
        # Check for message update from agent node
        if mode == "messages":
            message, metadata = chunk
            # Only process if from agent node and has content
            if (
                metadata.get("langgraph_node") == "agent"
                and hasattr(message, "content")
                and message.content
            ):
                content = message.content
                content_str = ""

                # Handle potential list content (multimodal)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, str):
                            content_str += block
                        elif isinstance(block, dict) and "text" in block:
                            content_str += block["text"]
                else:
                    content_str = str(content)

                if content_str:
                    sys.stdout.write(content_str)
                    sys.stdout.flush()
                    full_content += content_str

        # Check for status updates (e.g. tool calls)
        elif mode == "updates":
            step_name = list(chunk.keys())[0]
            if step_name == "agent":
                agent_messages = chunk["agent"].get("messages", [])
                if agent_messages:
                    last_msg = agent_messages[-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        tools = [t["name"] for t in last_msg.tool_calls]
                        sys.stdout.write(
                            f"\n[System: Using tools - {', '.join(tools)}]\nAgent: "
                        )
                        sys.stdout.flush()

    sys.stdout.write("\n")
    return full_content


def main():
    print("=========================================")
    print("       UniGo CLI Backend Interface       ")
    print("=========================================")
    print("Type 'exit' or 'quit' to close.")
    print("Type 'major' to switch to Major Recommendation mode (one-off).")
    print("=========================================")

    # In-memory history for this session
    # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    chat_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Major Recommendation Mode Debugging
        if user_input.lower() == "major":
            print("\n[Major Recommendation Mode]")
            print(
                "Enter onboarding answers JSON or plain text (will be treated as 'interests')."
            )
            print("For simple test, just type your interests.")

            major_input = input("Interests/Input: ").strip()
            if not major_input:
                continue

            # Mocking onboarding dictionary for simple CLI usage
            onboarding_answers = {
                "subjects": "",
                "interests": major_input,
                "career_goal": "",
                "strengths": "",
                "career_field": "",
            }

            print("\nAnalyzing...")
            try:
                result = run_major_recommendation(onboarding_answers)
                print("\n=== Recommendation Result ===")
                print(f"User Profile: {result.get('user_profile_text')}")
                print("Recommended Majors:")
                for major in result.get("recommended_majors", []):
                    print(
                        f"- {major.get('major_name')} (Score: {major.get('total_score')})"
                    )
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Normal Chat Mode
        try:
            # We use the stream function to get real-time output
            # backend.main.run_mentor_stream signature:
            # (question: str, chat_history: list[dict] | None = None, mode: str = "react", stream_mode: str | list[str] = "updates")
            generator = run_mentor_stream(
                question=user_input,
                chat_history=chat_history,
                mode="react",
                stream_mode=["messages", "updates"],
            )

            # Print response and capture full content for history
            response_content = print_stream(generator)

            # Update history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response_content})

        except Exception as e:
            print(f"\nError processing request: {e}")
            # Import traceback for detail debugging in CLI
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
