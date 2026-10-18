import os
import sys
import json
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scraper import fetch_website_links, fetch_website_contents

# =====================================================================
# 1. Environment & New Native Client Setup
# =====================================================================
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    sys.exit("Error: GEMINI_API_KEY not found. Please check your .env file.")

# En yeni Google GenAI istemcisini başlatıyoruz
client = genai.Client(api_key=api_key)
MODEL = "gemini-3.8-flash"

# =====================================================================
# 2. System Prompts 
# =====================================================================
link_system_prompt = """
You are an expert web researcher.
Extract relevant links pointing to key company pages (About, Products, Blog, Tech).
Respond strictly in JSON format: {"links": [{"type": "about", "url": "..."}]}
"""

brochure_system_prompt = """
You are a corporate communications writer. 
Create a professional Company Brochure in markdown (no code blocks).
Include: Overview, Products & Services, Target Audience, and Technology Highlights.
"""

# =====================================================================
# 3. Stage 1: Smart Link Filtering
# =====================================================================
def get_links_user_prompt(url: str) -> str:
    links = fetch_website_links(url)
    return f"Links found on {url}:\n" + "\n".join(links)


def select_relevant_links(url: str) -> dict:
    print(f"\n[1/3] Scanning and filtering links for: {url}...")
    
    # Yeni SDK ile JSON formatında veri çekme
    response = client.models.generate_content(
        model=MODEL,
        contents=get_links_user_prompt(url),
        config=types.GenerateContentConfig(
            system_instruction=link_system_prompt,
            response_mime_type="application/json",
        )
    )
    return json.loads(response.text)


# =====================================================================
# 4. Bridge: Data Aggregation & Truncation
# =====================================================================
def fetch_page_and_all_relevant_links(url: str) -> str:
    print(f"[2/3] Fetching landing page content for {url}...")
    landing_content = fetch_website_contents(url)
    
    links_data = select_relevant_links(url)
    link_list = links_data.get("links", [])
    
    print(f"Found {len(link_list)} relevant links. Fetching contents...")
    result = f"## Landing Page:\n\n{landing_content[:2000]}\n\n## Relevant Sub-pages:\n"

    for link in link_list:
        link_type = link.get("type", "page")
        link_url = link["url"]
        print(f"  -> Scraping content for '{link_type}' ({link_url})...")
        result += f"\n\n### Page Type: {link_type}\nURL: {link_url}\n"
        page_content = fetch_website_contents(link_url)
        result += page_content[:1000]

    return result


def get_brochure_user_prompt(company_name: str, url: str) -> str:
    extracted_data = fetch_page_and_all_relevant_links(url)
    return f"Build a brochure for: {company_name}\nData:\n{extracted_data}"[:6000]


# =====================================================================
# 5. Stage 2: Streaming Brochure Generation
# =====================================================================
def stream_brochure(company_name: str, url: str):
    print(f"\n[3/3] Generating company brochure ({MODEL} newest native SDK)...\n")
    print("=" * 60)
    
    # Yeni SDK ile daktilo efektli (stream) veri çekme
    response = client.models.generate_content_stream(
        model=MODEL,
        contents=get_brochure_user_prompt(company_name, url),
        config=types.GenerateContentConfig(
            system_instruction=brochure_system_prompt,
        )
    )

    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n" + "=" * 60)
    print("\nBrochure generation completed successfully.")


# =====================================================================
# 6. Command Line Interface
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Powered Automated Web Brochure Generator")
    parser.add_argument("--company", type=str, default="Hugging Face", help="Target company name")
    parser.add_argument("--url", type=str, default="https://huggingface.co", help="Target company landing page URL")
    
    args = parser.parse_args()
    stream_brochure(company_name=args.company, url=args.url)