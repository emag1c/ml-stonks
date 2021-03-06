import urllib.request
import urllib.parse
from urllib.error import HTTPError
from bs4 import BeautifulSoup as Soup
from time import time, sleep
from typing import Union, List
from datetime import datetime

GOOG_URL = "https://www.google.com/search"
SCRAPING_DOG_URL = "https://api.scrapingdog.com/scrape"
DEFAULT_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6)" \
             " AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15"
DATE_FMT = "%-m/%-d/%Y"


class BadStatusCode(Exception):
    def __init__(self, code):
        self.code = code


class GoogleNewsScrapingDog:

    def __init__(self, api_key="", lang="en", period="", start=None, end=None, ua=DEFAULT_UA):
        self.user_agent = ua
        self.headers = {'User-Agent': self.user_agent}
        self.__api_key = api_key
        self.__texts = []
        self.__links = []
        self.__results = []
        self.__lang = lang
        self.__period = period
        self.__start: Union[None, datetime] = start
        self.__end: Union[None, datetime] = end
        self.__url = f"{SCRAPING_DOG_URL}?api_key={self.__api_key}&url=" + GOOG_URL
        self.__exec_time = 0
        self.max_retries = 5
        self.retry_sleep = 0.1
        self.__search_url = ""
        self.__query = ""

    def set_lang(self, lang):
        self.__lang = lang

    def set_period(self, period):
        self.__period = period

    def set_time_range(self, start, end):
        self.__start = start
        self.__end = end

    def search(self, q):
        """
        Searches for a term in google news and retrieves the first page into __results.

        Parameters:
        key = the search term
        """
        self.__query = q
        self.scrape_page()

    def __retry(self, exception, page, attempts):
        if attempts >= self.max_retries:
            raise exception
        attempts += 1
        sleep_for = self.retry_sleep * attempts * attempts
        print(f'request for url: {self.__search_url} failed with error {exception}. retrying in {sleep_for} seconds')
        sleep(sleep_for)
        self.scrape_page(page, attempts)

    def scrape_page(self, page=1, attempts=0):
        """
        Retrieves a specific page from google news into __results.

        Parameter:
        page = number of the page to be retrieved
        """
        start = time()
        payload = {
            'q': self.__query,
            'lr': f'lang_{self.__lang}',
            'tbs': f"lr:lang_1{self.__lang}",
            'tbm': 'nws',
            'start': (10 * (page - 1)),
        }

        try:
            if self.__start is not None and self.__end is not None:
                payload['tbs'] += f",cdr:1,cd_min:{self.__start.strftime(DATE_FMT)}," \
                                  f"cd_max:{self.__end.strftime(DATE_FMT)}"
            elif self.__period != "":
                payload['tbs'] += f",qdr:{self.__period}"
        except AttributeError:
            raise AttributeError("You need to run a search() before using scrape_page().")

        self.__search_url = self.__url + "?" + urllib.parse.urlencode(payload)
        req = urllib.request.Request(self.__search_url, headers=self.headers)
        try:
            res = urllib.request.urlopen(req)
            page = res.read()
        except HTTPError as e:
            if e.code == 429 or e.code == 500:
                self.__retry(e, page, attempts)
            else:
                raise e
        try:
            content = Soup(page, "html.parser")
        except Exception as e:
            print(f"ERROR TRYING TO LOAD CONTENT: {e}")
            self.__retry(e, page, attempts)
            raise e
        try:
            result = content.find_all("div", id="search")[0].find_all("g-card")
        except IndexError:
            # no results were found
            print("no results found")
            return

        for item in result:
            try:
                tmp_text = item.find("div", {"role": "heading"}).text.replace("\n", "")
            except Exception:
                tmp_text = ''
            try:
                tmp_link = item.find("a").get("href")
            except Exception:
                tmp_link = ''
            try:
                tmp_media = item.findAll("g-img")[1].parent.text
            except Exception:
                tmp_media = ''
            try:
                tmp_date = item.find("div", {"role": "heading"}).next_sibling.findNext('div').findNext('div').text
            except Exception:
                tmp_date = ''
            try:
                tmp_desc = item.find("div", {"role": "heading"}).next_sibling.findNext('div').text.replace("\n", "")
            except Exception:
                tmp_desc = ''
            try:
                tmp_img = item.findAll("g-img")[0].find("img").get("src")
            except Exception:
                tmp_img = ''
            self.__texts.append(tmp_text)
            self.__links.append(tmp_link)
            self.__results.append(
                {'title': tmp_text,
                 'media': tmp_media,
                 'date': tmp_date,
                 'desc': tmp_desc,
                 'link': tmp_link,
                 'img': tmp_img})

        res.close()
        self.__exec_time = time() - start
        print(self.__exec_time)

    def result(self) -> List[dict]:
        """Returns the __results."""
        return self.__results

    def get_text(self) -> List[str]:
        """Returns only the __texts of the __results."""
        return self.__texts

    def get_links(self) -> List[str]:
        """Returns only the __links of the __results."""
        return self.__links

    def clear(self):
        self.__texts = []
        self.__links = []
        self.__results = []
        self.__exec_time = 0
