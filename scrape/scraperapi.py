from urllib import parse
from bs4 import BeautifulSoup as Soup
from time import time, sleep
from typing import Union, List
from datetime import datetime
from scraper_api import ScraperAPIClient

GOOG_URL = "https://www.google.com/search"
DATE_FMT = "%-m/%-d/%Y"


class BadStatusCode(Exception):
    def __init__(self, code):
        self.code = code


class GoogleNews:

    def __init__(self, key: str, lang="en", period="", ua=""):
        assert key != ""
        self.client = ScraperAPIClient(key)
        self.user_agent = ua
        self.__texts = []
        self.__titles = []
        self.__links = []
        self.__results = []
        self.__lang = lang
        self.__period = period
        self.__exec_time = 0

    def set_lang(self, lang):
        self.__lang = lang

    def search(self, q: Union[List[str], str], p: Union[List[int], int], start: datetime, end: datetime) -> List[dict]:
        """
        Searches for a term in google news and retrieves the first page into __results.

        Parameters:
        key = the search term
        """
        start_time = time()
        if isinstance(q, str):
            q = [q]
        if isinstance(p, int):
            p = [p]
        elif len(p) < 1:
            p = [1]

        for query in q:
            for page in p:
                out = self.scrape_page(query, page, start, end)
                for o in out:
                    if o["title"] not in self.__titles:
                        self.__results.append(o)
                        self.__links.append(o["link"])
                        self.__texts.append(o["title"] + " " + o["desc"])

        self.__exec_time = time() - start_time
        return self.__results

    def scrape_page(self, q: str, page: int, start: datetime, end: datetime, attempts=0):
        """
        page = number of the page to be retrieved
        """
        payload = {
            'q': q,
            'lr': f'lang_{self.__lang}',
            'tbs': f"lr:lang_1{self.__lang}",
            'tbm': 'nws',
            'start': (10 * (page - 1)),
        }

        out: List[dict] = []

        if start is not None and end is not None:
            payload['tbs'] += f",cdr:1,cd_min:{start.strftime(DATE_FMT)}," \
                              f"cd_max:{end.strftime(DATE_FMT)}"

        try:
            page = self.client.get(url=GOOG_URL + "?" + parse.urlencode(payload)).text
            content = Soup(page, "html.parser")
        except Exception as e:
            attempts += 1
            if attempts > 5:
                print(f"ERROR TRYING TO LOAD CONTENT: {e}")
                raise e
            sleep(0.1 * attempts)
            self.scrape_page(q, page, start, end, attempts)
        try:
            result = content.find_all("div", id="search")[0].find_all("g-card")
        except IndexError:
            # no results were found
            return out

        for item in result:
            try:
                out.append({
                    "title": item.find("div", {"role": "heading"}).text.replace("\n", ""),
                    "link": item.find("a").get("href"),
                    "media": item.findAll("g-img")[1].parent.text,
                    "date": item.find("div", {"role": "heading"}).next_sibling.findNext('div').findNext('div').text,
                    "desc": item.find("div", {"role": "heading"}).next_sibling.findNext('div').text.replace("\n", ""),
                    "image": item.findAll("g-img")[0].find("img").get("src")
                })
            except Exception:
                pass
        return out

    def get_results(self) -> List[dict]:
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
        self.__titles = []
        self.__exec_time = 0
