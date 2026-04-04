import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

class BrowserManager:
    """Manages the Playwright browser lifecycle and provides high-level tools."""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def start(self):
        """Start the browser in non-headless mode for visibility."""
        if not self.playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=False)
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()
        return self.page

    async def stop(self):
        """Close the browser and cleanup."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def navigate(self, url: str) -> str:
        """Navigates to a URL and returns the page title."""
        if not self.page:
            await self.start()
        
        try:
            await self.page.goto(url, wait_until="networkidle", timeout=30000)
            title = await self.page.title()
            return f"Successfully navigated to: {url}\nPage Title: {title}"
        except Exception as e:
            return f"Error navigating to {url}: {e}"

    async def get_page_content(self) -> str:
        """Extracts and truncates the page content (skeleton markdown for efficiency)."""
        if not self.page:
            return "Error: No page is currently open. Navigate first."
        
        try:
            html = await self.page.content()
            soup = BeautifulSoup(html, "html.parser")
            
            # Aggressive cleaning for token efficiency
            for element in soup(["script", "style", "nav", "footer", "aside", "header"]):
                element.extract()
            
            # Focusing on buttons, links, and headings (the 'skeleton')
            for tag in soup.find_all(['h1', 'h2', 'h3', 'button', 'a']):
                tag.insert_before("\n")
                tag.insert_after("\n")

            text = soup.get_text(separator=" ")
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)
            
            # Truncate to 3000 chars for token efficiency
            truncated_text = clean_text[:3000]
            if len(clean_text) > 3000:
                truncated_text += "\n... [Content Truncated]"
                
            return f"--- PAGE SKELETON ---\n{truncated_text}"
        except Exception as e:
            return f"Error extracting page content: {e}"

    async def get_links(self) -> str:
        """Returns the top 20 most relevant clickable links on the page."""
        if not self.page:
            return "Error: No page open."
        
        try:
            links = await self.page.query_selector_all("a")
            results = []
            for link in links[:20]: 
                text = await link.inner_text()
                href = await link.get_attribute("href")
                if text.strip() and href:
                    results.append(f"- [{text.strip()}]({href})")
            
            return "--- TOP 20 LINKS ---\n" + "\n".join(results) if results else "No links found."
        except Exception as e:
            return f"Error getting links: {e}"

    async def fill_form(self, selector_or_text: str, value: str) -> str:
        """Fills an input field with the given value."""
        if not self.page:
            return "Error: No page open."
        
        try:
            # Try to find by placeholder or label text first
            try:
                await self.page.get_by_placeholder(selector_or_text).first.fill(value, timeout=5000)
                return f"Filled field '{selector_or_text}' with value."
            except:
                try:
                    await self.page.get_by_label(selector_or_text).first.fill(value, timeout=5000)
                    return f"Filled field with label '{selector_or_text}'."
                except:
                    # Fallback to CSS
                    await self.page.fill(selector_or_text, value, timeout=5000)
                    return f"Filled selector '{selector_or_text}'."
        except Exception as e:
            return f"Error filling form: {e}"

    async def scroll(self, direction: str = "down") -> str:
        """Scrolls the page up or down."""
        if not self.page:
            return "Error: No page open."
        
        try:
            amount = 500 if direction == "down" else -500
            await self.page.evaluate(f"window.scrollBy(0, {amount})")
            return f"Scrolled {direction}."
        except Exception as e:
            return f"Error scrolling: {e}"

    async def take_screenshot(self, filename: str = "screenshot.png") -> str:
        """Takes a screenshot of the current page and saves it to the workspace."""
        if not self.page:
            return "Error: No page open."
        
        try:
            path = os.path.join(os.getcwd(), "workspace", filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            await self.page.screenshot(path=path)
            return f"Screenshot saved at {filename}. Use this to visualize the layout if you are stuck."
        except Exception as e:
            return f"Error taking screenshot: {e}"

    async def click_element(self, text_or_selector: str) -> str:
        """Clicks an element by visible text or selector."""
        if not self.page:
            return "Error: No page is open."
        
        try:
            # Attempt to click by text first 
            try:
                await self.page.get_by_text(text_or_selector).first.click(timeout=5000)
                return f"Successfully clicked element with text: '{text_or_selector}'"
            except:
                # Fallback to CSS selector
                await self.page.click(text_or_selector, timeout=5000)
                return f"Successfully clicked element with selector: '{text_or_selector}'"
        except Exception as e:
            return f"Error clicking element '{text_or_selector}': {e}"
