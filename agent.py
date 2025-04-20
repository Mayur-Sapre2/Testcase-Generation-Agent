import re
import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_groq.chat_models import ChatGroq
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, dotenv_values 

load_dotenv()

tavily_client = TavilyClient()

## Define the GraphState 
class GraphState(TypedDict):
    user_request: str
    requirements_docs_content: str
    requirements_docs_summary: str
    testcases_format: str
    testcases: str
    answer: str
    tavily_search_results: str
    industry_best_practices: str
    enhanced_testcases: str

## To generate Summary of the requirements document
def generate_summary_node_function(state: GraphState) -> GraphState:
    requirements_docs_content = state.get("requirements_docs_content", "")
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    prompt = (
        "You are an expert in generating QA testcases for any known formats. \n" + 
        "Study the given 'Requirements Documents Content' carefully and generate summary of about 5 lines\n" +
        f"Requirements Documents Content: {requirements_docs_content}\n" +
        "Answer:"
    )
    
    try:
        response = st.session_state.llm.invoke(prompt)
    except Exception as e:
        response = f"Error generating answer: {str(e)}"
        
    state['requirements_docs_summary'] = response.content
    state['answer'] = response.content
    return state

## Search for industry best practices using Tavily
def search_best_practices_node_function(state: GraphState) -> GraphState:
    summary = state.get("requirements_docs_summary", "")
    user_request = state.get("user_request", "")
    testcases_format = state.get("testcases_format", "")
    
    # Create search query based on requirements summary and test format
    search_query = f"Best practices for testing {summary} using {testcases_format} test format"
    
    try:
        # Perform Tavily search
        search_results = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=3
        )
        
        # Extract and format the search results
        formatted_results = "INDUSTRY BEST PRACTICES:\n\n"
        for result in search_results.get("results", []):
            formatted_results += f"- {result.get('title', 'No title')}\n"
            formatted_results += f"  {result.get('content', 'No content')[:300]}...\n\n"
        
        state["tavily_search_results"] = formatted_results
        
        # Use LLM to extract best practices from search results
        prompt = (
            "You are an expert in software testing. Based on the following search results about best practices, "
            "provide a concise summary of the most relevant test practices that would apply to our specific requirements. "
            f"Focus on {testcases_format} test format.\n\n"
            f"SEARCH RESULTS:\n{formatted_results}\n\n"
            f"REQUIREMENTS SUMMARY:\n{summary}\n\n"
            "Provide 3-5 best practices that we should follow when creating our test cases:"
        )
        
        best_practices_response = st.session_state.llm.invoke(prompt)
        state["industry_best_practices"] = best_practices_response.content
        
    except Exception as e:
        state["tavily_search_results"] = f"Error performing search: {str(e)}"
        state["industry_best_practices"] = "Could not retrieve industry best practices."
    
    return state

## Router function to decide whether to output gherkin or selenium
def route_user_request(state: GraphState) -> str:
    user_request = state["user_request"]
    tool_selection = {
    "gherkin_format": (
        "Use requests generation of testcases in Gherkin format "
    ),
    "selenium_format": (
        "Use requests generation of testcases in Selenium format"
    )
    }

    SYS_PROMPT = """Act as a router to select specific testcase format or functions based on user's request, using the following rules:
                    - Analyze the given user's request and use the given tool selection dictionary to output the name of the relevant tool based on its description and relevancy. 
                    - The dictionary has tool names as keys and their descriptions as values. 
                    - Output only and only tool name, i.e., the exact key and nothing else with no explanations at all. 
                """

    # Define the ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            ("human", """Here is the user's request:
                        {user_request}
                        Here is the tool selection dictionary:
                        {tool_selection}
                        Output the required tool name from the tool selection dictionary only. Just one word.
                    """),
        ]
    )

    # Pass the inputs to the prompt
    inputs = {
        "user_request": user_request,
        "tool_selection": tool_selection
    }

    # Invoke the chain
    tool = (prompt | st.session_state.llm | StrOutputParser()).invoke(inputs)
    tool = re.sub(r"[\\'\"`]", "", tool.strip())  

    if "gherkin" in tool:
        tool = "gherkin"
    else:
        tool = "selenium"
        
    state["testcases_format"] = tool
    return tool

def generate_testcases(user_request, requirements_content, llm, format_type, best_practices=""):
    prompt = (
        "You are an expert in generating QA testcases for any known formats. \n" + 
        "Study the given 'Requirements Documents Content' carefully and generate about 3 testcases in the suggested 'Format'\n" +
        "You may want to look at the original User Request just to make sure that you are answering the request properly.\n" +
        f"User Request: {user_request}\n" +
        f"Requirements Documents Content: {requirements_content}\n" +
        f"Format: {format_type}\n"
    )
    
    if best_practices:
        prompt += f"\nIndustry Best Practices to Follow:\n{best_practices}\n"
    
    prompt += "\nAnswer:"
    
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        response = f"Error generating answer: {str(e)}"
        
    return response.content

## To generate Gherkin formatted Testcases
def generate_gherkin_testcases_node_function(state: GraphState) -> GraphState:
    user_request = state["user_request"]
    requirements_docs_content = state.get("requirements_docs_content", "")
    testcases_format = state.get("testcases_format", "gherkin")
    best_practices = state.get("industry_best_practices", "")
    
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    response = generate_testcases(user_request, requirements_docs_content, st.session_state.llm, testcases_format, best_practices)
    
    state['testcases'] = response
    state['answer'] = response
    return state

## To generate Selenium formatted Testcase
def generate_selenium_testcases_node_function(state: GraphState) -> GraphState:  
    user_request = state["user_request"]
    requirements_docs_content = state.get("requirements_docs_content", "")
    testcases_format = state.get("testcases_format", "selenium")
    best_practices = state.get("industry_best_practices", "")
    
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    response = generate_testcases(user_request, requirements_docs_content, st.session_state.llm, testcases_format, best_practices)
    
    state['testcases'] = response
    state['answer'] = response
    return state

## Enhance test cases with examples and edge cases
def enhance_testcases_node_function(state: GraphState) -> GraphState:
    testcases = state.get("testcases", "")
    requirements_summary = state.get("requirements_docs_summary", "")
    testcases_format = state.get("testcases_format", "")
    search_query = f"Common edge cases and test examples for {testcases_format} tests in e-commerce applications"
    
    try:
        # Perform Tavily search for edge cases
        edge_case_results = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=2
        )
        
        # Extract and format the search results
        edge_cases_content = "EDGE CASES AND TEST EXAMPLES:\n\n"
        for result in edge_case_results.get("results", []):
            edge_cases_content += f"- {result.get('title', 'No title')}\n"
            edge_cases_content += f"  {result.get('content', 'No content')[:250]}...\n\n"
        
        # Use LLM to enhance test cases with edge cases
        prompt = (
            "You are an expert in software testing. You've created some initial test cases, "
            "but now you need to enhance them by adding edge cases and improving coverage.\n\n"
            f"INITIAL TEST CASES:\n{testcases}\n\n"
            f"REQUIREMENTS SUMMARY:\n{requirements_summary}\n\n"
            f"EDGE CASE INFORMATION:\n{edge_cases_content}\n\n"
            f"Please enhance the {testcases_format} test cases by:\n"
            "1. Adding at least 2 edge case scenarios\n"
            "2. Including boundary conditions\n"
            "3. Considering negative testing scenarios\n"
            "4. Adding performance-related tests if applicable\n"
            "Maintain the same format as the original test cases."
        )
        
        enhanced_response = st.session_state.llm.invoke(prompt)
        state["enhanced_testcases"] = enhanced_response.content
        state["answer"] = enhanced_response.content
        
    except Exception as e:
        state["answer"] = testcases  # Fall back to original test cases
        print(f"Error enhancing test cases: {str(e)}")
    
    return state

## Build the LangGraph pipeline
def build_workflow():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("summary_node", generate_summary_node_function)
    workflow.add_node("best_practices_node", search_best_practices_node_function)
    workflow.add_node("gherkin_node", generate_gherkin_testcases_node_function)
    workflow.add_node("selenium_node", generate_selenium_testcases_node_function)
    workflow.add_node("enhance_node", enhance_testcases_node_function)
    
    # Set entry point and add edges
    workflow.set_entry_point("summary_node")
    workflow.add_edge("summary_node", "best_practices_node")
    
    # Add conditional edges based on test format
    workflow.add_conditional_edges(
        "best_practices_node",
        route_user_request,  # Router function
        {
            "gherkin": "gherkin_node",
            "selenium": "selenium_node"
        }
    )
    
    # Add enhancement step after test case generation
    workflow.add_edge("gherkin_node", "enhance_node")
    workflow.add_edge("selenium_node", "enhance_node")
    workflow.add_edge("enhance_node", END)
    
    return workflow

## The initialize_app function
def initialize_app(model_name: str):
    """
    Initialize the app with the given model name, avoiding redundant initialization.
    """
    # Check if the LLM is already initialized
    if "selected_model" in st.session_state and st.session_state.selected_model == model_name:
        return build_workflow().compile()  # Return the compiled workflow

    # Initialize the LLM for the first time or switch models
    st.session_state.llm = ChatGroq(model=model_name, temperature=0.0)
    st.session_state.selected_model = model_name
    print(f"Using model: {model_name}")
    return build_workflow().compile()