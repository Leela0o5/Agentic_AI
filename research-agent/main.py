import sys
from argparse import ArgumentParser
from src.agent import ResearchAgent


def main():
    parser = ArgumentParser(
        description="Research Agent — ask it a question, it researches and remembers"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="The research question to investigate",
    )
    parser.add_argument(
        "--clear-memory",
        action="store_true",
        help="Clear all stored memory and start fresh",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show memory status without running research",
    )
    
    args = parser.parse_args()
    
    try:
        agent = ResearchAgent()
        
        if args.clear_memory:
            agent.clear_memory()
            return
        
        if args.status:
            print(agent.get_memory_status())
            return
        
        if not args.query:
            parser.print_help()
            print("\nExample: python main.py 'what are large language models?'")
            return
        
        # Run the research
        result = agent.research(args.query)
        print("\n" + "="*70)
        print("RESEARCH FINDINGS")
        print("="*70)
        print(result)
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Research cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
