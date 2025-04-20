# Test Case Generation Agent

## Overview

The Test Case Generation Agent is an AI-powered application that automatically generates high-quality test cases from requirements documents. It leverages advanced LLMs and search capabilities to produce comprehensive test suites in either Gherkin or Selenium format, incorporating industry best practices and edge cases.

![demo](demo/demo.gif)

## Features

- **Automated Test Case Generation** from plain text or PDF requirements documents
- **Multiple Test Formats** including Gherkin and Selenium
- **Industry Standards Integration** using Tavily search to incorporate best practices
- **Edge Case Detection** to ensure comprehensive test coverage
- **Customizable Detail Level** to match your testing needs
- **Interactive Web Interface** built with Streamlit

## Installation

1. Clone this repository
   ```bash
   git clone https://github.com/codemaker2015/testcase-generation-agent.git
   cd testcase-generation-agent
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys by creating a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

1. Start the application
   ```bash
   streamlit run app.py
   ```

2. Access the web interface at `http://localhost:8501`

3. Upload a requirements document (TXT or PDF)

4. Configure test generation settings in the sidebar:
   - Select your preferred LLM model
   - Choose test format preferences
   - Adjust detail level and edge case inclusion

5. Enter a request like "Generate Gherkin test cases for the login feature" and press Enter

## Example Prompts

- "Create Gherkin test cases for the cart functionality"
- "Generate Selenium tests for the checkout process"  
- "Develop test cases with edge cases for payment processing"
- "Create tests focusing on performance requirements"

## Components

- **app.py**: The main Streamlit web application
- **agent.py**: LangGraph workflow implementation for test generation
- **requirements.txt**: Required Python packages

## LLM Models Supported

- llama-3.1-8b-instant
- llama-3.3-70b-versatile
- llama3-70b-8192
- llama3-8b-8192
- mixtral-8x7b-32768
- gemma2-9b-it

## Advanced Configuration

The application allows for several advanced configurations:

1. **Edge Case Detection**: Enable or disable automatic identification of edge cases
2. **Detail Level**: Adjust the comprehensiveness of generated test cases
3. **Industry Standards**: Toggle whether to incorporate industry best practices
4. **Default Test Format**: Pre-select a test format or use auto-detection

## Sample Requirements Document

The repository includes a sample requirements document (`input.txt`) that you can use to test the application. This document describes an e-commerce shopping cart system with detailed functional and non-functional requirements.

## Dependencies

- streamlit
- langgraph
- langchain
- tavily-python
- groq
- python-dotenv
- pypdf

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.