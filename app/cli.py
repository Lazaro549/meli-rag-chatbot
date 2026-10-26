"""Simple CLI entry point."""

import argparse
import sys

from app.agents import run_agent


def main():
    parser = argparse.ArgumentParser(description="MELI RAG Chatbot CLI")
    parser.add_argument("query", nargs="?", help="Question to ask the assistant")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive or not args.query:
        print("MELI RAG Chatbot – interactive mode (type 'exit' to quit)\n")
        history = []
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not query or query.lower() in {"exit", "quit", "q"}:
                print("Bye!")
                break
            result = run_agent(query, chat_history=history)
            print(f"\nAssistant: {result['answer']}\n")
            # Keep simple history (Human + AI)
            from langchain_core.messages import AIMessage, HumanMessage
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=result["answer"]))
    else:
        result = run_agent(args.query)
        print(result["answer"])


if __name__ == "__main__":
    main()
