from typing import Optional
import google.genai as genai

from .memory import ResearchMemory
from .tools import WebSearchTool, PageFetchTool
from . import config


class ResearchAgent:

    def __init__(self):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.GEMINI_MODEL
        
        self.memory = ResearchMemory()
        self.search_tool = WebSearchTool()
        self.fetch_tool = PageFetchTool()

    def research(self, query: str) -> str:
        """Execute a research task."""
        validation = self._validate_query(query)
        if validation:
            print(f"\n{validation}\n")
            return validation
        
        print(f"\n[RESEARCH] Starting research on: {query}\n")
        
        print("[RECALL] Checking memory for past findings...")
        past_findings = self.memory.recall(query, top_k=3)
        
        if past_findings:
            print(f"[RECALL] Found {len(past_findings)} relevant findings:")
            for finding in past_findings:
                print(f"  • From {finding['source']} (similarity: {finding['similarity']:.2f})")
        else:
            print("[RECALL] No past findings. Starting fresh.")
        
        print("\n[SEARCH] Searching the web...")
        search_results = self.search_tool.search(query, max_results=5)
        print(f"[SEARCH] Found {len(search_results)} results")
        
        print("\n[FETCH] Reading relevant pages...")
        fetched_content = []
        for result in search_results[:3]:  
            url = result.get("url")
            print(f" Fetching {url}...")
            
            content = self.fetch_tool.fetch_and_extract(url)
            if content:
                fetched_content.append({
                    "url": url,
                    "title": result.get("title", ""),
                    "content": content,
                })
                print(f"   Extracted {len(content)} characters")
            else:
                print(f"   Failed to fetch")
        
        print("\n[ANALYZE] Synthesizing findings with LLM...")
        
        context = self._build_prompt(query, past_findings, fetched_content, search_results)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=context,
            config=genai.types.GenerateContentConfig(
                temperature=config.TEMPERATURE,
                max_output_tokens=config.MAX_TOKENS_FOR_RESPONSE,
            ),
        )
        
        analysis = response.text
        print("[ANALYZE] LLM synthesis complete")
        
        print("\n[SAVE] Storing findings in memory...")
        for content_obj in fetched_content:
            self.memory.save(
                content=content_obj["content"][:1000],  
                source=content_obj["url"],
                topic=query,
            )
        print(f"[SAVE] Saved {len(fetched_content)} findings to memory")
        
        result = self._format_response(analysis, past_findings, fetched_content)
        return result

    def _build_prompt(
        self,
        query: str,
        past_findings: list[dict],
        fetched_content: list[dict],
        search_results: list[dict],
    ) -> str:
        """Build the LLM prompt with context."""
        system_instruction = """You are a research assistant. Your ONLY role:
- Answer research questions about topics, frameworks, concepts, technologies, trends
- Synthesize information from web sources and past research
- Provide citations for all claims

You DO NOT:
- Answer non-research questions (jokes, entertainment, personal favors, creative writing)
- Provide opinions or take sides on non-technical matters
- Give advice outside research scope (dating, health, financial, legal)
- Pretend to have capabilities you don't (coding for users, hacking, etc)

If the question is NOT research-related, refuse clearly and redirect to research topics."""
        
        prompt = f"""{system_instruction}

You are a research assistant synthesizing information about: {query}

PAST FINDINGS (from previous research sessions):
"""
        if past_findings:
            for i, finding in enumerate(past_findings, 1):
                prompt += f"\n{i}. From {finding['source']} (similarity: {finding['similarity']:.2%}):\n"
                prompt += f"   {finding['content'][:500]}\n"
        else:
            prompt += "   (No previous research on this topic)\n"

        prompt += "\nNEW SEARCH RESULTS:\n"
        for i, result in enumerate(search_results, 1):
            prompt += f"\n{i}. {result['title']}\n"
            prompt += f"   URL: {result['url']}\n"
            prompt += f"   Snippet: {result['snippet']}\n"

        prompt += "\nFETCHED PAGE CONTENT:\n"
        for content_obj in fetched_content:
            prompt += f"\n--- From {content_obj['title']} ---\n"
            prompt += content_obj["content"][:2000]
            prompt += "\n"

        prompt += f"""
TASK: 
1. Review the past findings and new information
2. Identify key insights that answer the research question
3. Note any conflicting information or updates from previous sessions
4. Cite your sources (include URLs)
5. Keep your answer concise and directly relevant to the query

Research Question: {query}

Provide a well-researched answer:"""
        
        return prompt

    def _format_response(
        self,
        analysis: str,
        past_findings: list[dict],
        fetched_content: list[dict],
    ) -> str:
        """Format the final response with source information."""
        response = analysis
        
        if past_findings:
            response += "\n\n--- From Memory (Previous Sessions) ---\n"
            for finding in past_findings:
                response += f"• {finding['source']}\n"
        
        if fetched_content:
            response += "\n\n--- Sources (This Session) ---\n"
            for content_obj in fetched_content:
                response += f"• {content_obj['url']}\n"
        # Add memory statistics
        stats = self.memory.get_statistics()
        response += f"\n\n--- Memory Status ---\n"
        response += f"Total findings stored: {stats['total_findings']}\n"
        
        return response

    def clear_memory(self) -> None:
        """Delete all stored memory and start fresh."""
        print("[WARNING] Clearing all memory...")
        self.memory.clear()
        print("[OK] Memory cleared. Agent reset to newborn state.")

    def get_memory_status(self) -> str:
        """Get a summary of what the agent has learned so far."""
        stats = self.memory.get_statistics()
        return f"""
Memory Status:
  Total findings: {stats['total_findings']}
  Storage location: {stats['memory_path']}
"""

    def _validate_query(self, query: str) -> Optional[str]:
        """Check if query is serious research. Return error message if not."""
        if not query or not query.strip():
            return "[ERROR] Empty query. Please ask something meaningful."
        
        query = query.strip()
        
        if len(query) < 3:
            return "[ERROR] Query too short. Ask a proper question."
        
        if query.count("?") > 3 or query.count("!") > 3:
            return "[ERROR] That's not a serious question. Please ask something substantive."
        
        if query.lower() in ["yes", "no", "ok", "lol", "test", "hi", "hey", "what", "why", "how"]:
            return "[ERROR] Be more specific. 'What' or 'Why' alone isn't a research question."
        
        if all(c in "?!. " for c in query):
            return "[ERROR] That's not a research question."
        
        if query.lower().startswith(("aaaaaa", "xxxxx", "123", "asdf", "qwerty")):
            return "[ERROR] That looks like random input. Ask something real."
        
        if len(query.split()) == 1 and query[0].isalpha():
            return "[ERROR] Single words don't count as research. Phrase a real question."
        
        # Reject non-research requests
        non_research_phrases = [
            "tell me a joke", "sing", "write a poem", "make me", "do my homework",
            "play a game", "what's the weather", "schedule", "remind me", "set a timer",
            "cook", "recipe", "workout", "exercise routine", "dating advice",
            "astrology", "your opinion", "roast me", "compliment", "pickup lines",
            "write code for me", "hack", "jailbreak"
        ]
        
        query_lower = query.lower()
        for phrase in non_research_phrases:
            if phrase in query_lower:
                return "[ERROR] That's not a research question. I research topics, not provide entertainment or personal favors."
        
        # Check for at least some research-oriented intent
        research_keywords = [
            "what", "how", "why", "explain", "difference", "comparison", "trend",
            "history", "overview", "guide", "architecture", "design", "pattern",
            "framework", "library", "technology", "concept", "analysis", "research",
            "study", "learn", "understand", "background", "definition", "guide to",
            "introduction to", "vs", "vs.", "between"
        ]
        
        if not any(keyword in query_lower for keyword in research_keywords):
            return "[ERROR] That doesn't look like a research question. Ask about a topic, framework, concept, or trend."
        
        return None
