from Agent import iterative_rag_agent

def main():
    print("🔍 Welcome to the RAG Assistant\n")
    
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break

        result = iterative_rag_agent(user_query)

        print("\n🧠 Assistant Response:")
        print(result["answer"])

        print("\n📚 References:")
        for ref in result["references"]:
            print("-", ref)

        if result.get("needs_refinement"):
            print("\n⚠️ The assistant suggests refining your query.")
        # else:
        #     print("\n📚 References:")
        #     for ref in result["references"]:
        #         print("-", ref)

        print("\n" + "="*50 + "\n")
        
                # 🧩 Debug: show chunks that were retrieved
        print("\n📄 Retrieved Chunks:")
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"\n--- Chunk {i} ---")
            print(chunk[:500])  # Limit output if chunks are long

if __name__ == "__main__":
    main()
