import sys
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import json
from bs4 import BeautifulSoup

class Quotes(CrawlSpider):  # Change to CrawlSpider
    name = 'quotes'
    allowed_domains = ['quotes.toscrape.com']  # Specify allowed domain

    rules = (
        Rule(LinkExtractor(allow=r'/author'), callback='parse_page', follow=True),  # existing rule
        Rule(LinkExtractor(allow=r'/page/\d+/', restrict_css='li.next'), follow=True),  # new rule for "Next" button
    )
    
    def __init__(self, start_url='https://quotes.toscrape.com/', depth_limit=2, page_count=10, *args, **kwargs):
        super(Quotes, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.custom_settings = {
            'DEPTH_LIMIT': depth_limit,
            'CLOSESPIDER_PAGECOUNT': page_count,
            'DEPTH_PRIORITY': 1,
            'CONCURRENT_REQUESTS':32,
            'HTTPCACHE_ENABLED': True,
            'DOWNLOAD_DELAY': 0.25,
            'SCHEDULER_DISK_QUEUE': 'scrapy.squeues.PickleFifoDiskQueue',
            'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue',
        }
        self.data = []
        # Print the configurations when starting
        print(f"Starting URL: {start_url}")
        print(f"Depth Limit: {depth_limit}")
        print(f"Page Count: {page_count}")
        print(f"Custom Settings: {self.custom_settings}")

    # Removed parse method as CrawlSpider uses rules to follow links
    
    def parse_page(self, response):  # This will now only parse pages with /author
        self.log('Visited %s' % response.url)
        soup = BeautifulSoup(response.text, 'html.parser')

        footer = soup.find('footer')
        if footer:
            footer.decompose()

        for text_node in soup.find_all(text=True):
            text = text_node.strip()
            word_count = len(text.split())
            if word_count < 4:
                text_node.extract()

        page_text = soup.get_text(separator=' ', strip=True)
        self.data.append({
            'url': response.url,
            'text': page_text
        })

    def closed(self, reason):
        with open('data/output.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })
    start_url='https://quotes.toscrape.com/'
    depth_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    page_count = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    process.crawl(Quotes, start_url=start_url, depth_limit=depth_limit, page_count=page_count)
    process.start()