import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def test_llm_processing():
    # Load environment variables
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in .env file")
        return

    try:
        # Initialize Groq
        llm = ChatGroq(
            model="gemma2-9b-it",
            groq_api_key=GROQ_API_KEY
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that takes a list of sign language words "
                      "and converts them into a natural, grammatically correct sentence. "
                      "Focus on maintaining the original meaning while making it flow naturally."),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | llm

        # Test cases - simulating collected words from sign language
        test_cases = [
            ["I", "Love", "Python", "Programming"],
            ["Hello", "How", "Are", "You"],
            ["Thank", "You", "Very", "Much"],
            ["She", "Going", "School"]  # Intentionally grammatically incorrect
        ]

        print("\nTesting LLM sentence processing...")
        print("-" * 50)
        
        for words in test_cases:
            print(f"\nInput words: {' '.join(words)}")
            try:
                response = chain.invoke({
                    "messages": [
                        HumanMessage(content=f"Convert these sign language words into a natural sentence: {' '.join(words)}")
                    ]
                })
                print(f"Processed sentence: {response.content}")
                print("-" * 30)
            except Exception as e:
                print(f"Error processing sentence: {e}")

    except Exception as e:
        print(f"Error initializing LLM: {e}")

if __name__ == "__main__":
    test_llm_processing()