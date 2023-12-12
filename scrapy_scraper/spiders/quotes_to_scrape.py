import sys
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import json
from bs4 import BeautifulSoup
from os.path import join
from urllib.parse import urljoin


class Quotes(CrawlSpider):
    name = 'quotes'
    allowed_domains = ["quotes.toscrape.com"]

    rules = (
        Rule(LinkExtractor(allow=r'/author'), callback='parse_page', follow=True),
        Rule(LinkExtractor(allow=r'/page/\d+/', restrict_css='li.next'), callback='parse_quotes', follow=True),
    )

    def __init__(self, start_url='https://quotes.toscrape.com/page/1', depth_limit=2, page_count=10, *args, **kwargs):
        super(Quotes, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.custom_settings = {
            'DEPTH_LIMIT': depth_limit,
            'CLOSESPIDER_PAGECOUNT': page_count,
            'DEPTH_PRIORITY': 1,
            'CONCURRENT_REQUESTS': 32,
            'HTTPCACHE_ENABLED': True,
            'DOWNLOAD_DELAY': 0.25,
            'SCHEDULER_DISK_QUEUE': 'scrapy.squeues.PickleFifoDiskQueue',
            'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue',
        }
        self.url2data = {}
        self.BASE_URL = "http://quotes.toscrape.com"

        print(f"Starting URL: {start_url}")
        print(f"Depth Limit: {depth_limit}")
        print(f"Page Count: {page_count}")
        print(f"Custom Settings: {self.custom_settings}")

        self.url_set = set()

    def parse_quotes(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        quotes = soup.find_all('div', class_='quote')
        author_links = soup.find_all('a', href=True, text='(about)')

        for quote, author_link in zip(quotes,author_links):
            url = author_link["href"] + "/"
            url = urljoin(self.BASE_URL, url)

            text = quote.find('span', itemprop='text').get_text()
            if "quote" in self.url2data.get(url, {}):
                self.url2data[url]["quote"].append(text)
            else:
                self.url2data[url] = self.url2data.get(url, {})
                self.url2data[url]["quote"] = [text]


        self.url_set.add(url)

    def parse_page(self, response):
        self.log('Visited %s' % response.url)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text(separator=' ', strip=True)
        page_text = page_text.split("Quotes by:")[0]

        if response.url not in self.url2data:
            self.url2data[response.url] = {}
        self.url2data[response.url]["url"] = response.url
        self.url2data[response.url]["description"] = page_text

        self.url_set.add(response.url)

    def closed(self, reason):
        sorted_data = [{key: value[key] for key in sorted(value, reverse=True)} for value in self.url2data.values()]
        #data = list(self.url2data.values())
        output_path = join('data', 'output.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })
    start_url = 'https://quotes.toscrape.com/'
    depth_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    page_count = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    process.crawl(Quotes, start_url=start_url, depth_limit=depth_limit, page_count=page_count)
    process.start()
