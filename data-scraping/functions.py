import numpy as np
import pandas as pd
import re
import tqdm
import requests
from bs4 import BeautifulSoup
import time

def scrape_info_from_one_film(one_film):
    
    try:
        title = one_film.find('a').text
    except AttributeError:
        title = np.nan
    
    try:
        link = 'https://www.imdb.com' + one_film.find('a').get('href')
    except AttributeError:
        link = np.nan
    
    try:
        release_date = one_film.find('span', class_='lister-item-year text-muted unbold').text
        release_date = int(re.sub('[^0-9]', '', release_date)) 
    except (AttributeError, ValueError):
        release_date = np.nan
    
    try:
        duration = one_film.find('span', class_='runtime').text
        duration = re.sub(' min', '', duration)
        duration = int(re.sub(',', '', duration)) 
    except AttributeError:
        duration = np.nan
    
    try:
        genre = one_film.find('span', class_='genre').text
        genre = re.sub('\n', '', genre).strip()
        genre = [i for i in genre.split(', ')]
        genre = ', '.join(genre)
    except AttributeError:
        genre = np.nan
    
    try:
        imdb_rating = one_film.find('strong').text
        imdb_rating = float(imdb_rating)
    except AttributeError:
        imdb_rating = np.nan
    
    try:
        metascore = one_film.find('div', class_='inline-block ratings-metascore').find('span').text
        metascore = int(metascore.strip())
    except AttributeError:
        metascore = np.nan
    
    try:        
        synopsis = one_film.find_all('p', class_='text-muted')[1].text
        synopsis = re.sub('\n', '', synopsis).strip()
    except (AttributeError, IndexError):
        synopsis = np.nan
    
    try:
        director = one_film.find('p', class_='').find('a').text
    except AttributeError:
        director = np.nan
    
    try:   
        actors = one_film.find('p', class_='').find_all('a')[1:]
        actors = [i.text for i in actors]
    except (AttributeError, IndexError):
        actors = np.nan
    
    try:
        votes = one_film.find('p', class_='sort-num_votes-visible').text
        votes = int(re.sub(',', '', votes.split('\n')[2]))
    except(AttributeError, IndexError, ValueError):
        votes = np.nan
    
    one_film_dict = {
        'title':title,
        'link':link,
        'release_date':release_date,
        'duration':duration,
        'genre':genre,
        'imdb_rating':imdb_rating,
        'metascore':metascore,
        'synopsis':synopsis,
        'director':director,
        'actors':actors,
        'votes':votes
    }
    return one_film_dict


def scrape_info_from_one_page(one_page):
    one_page_df = pd.DataFrame()
    for i in range(len(one_page)):
        tmp_film = scrape_info_from_one_film(one_page[i])
        one_page_df = one_page_df.append(tmp_film, ignore_index=True)
    return one_page_df


def scrape_info_from_all_pages(links_to_scrape):
    all_films = pd.DataFrame()
    for link in range(links_to_scrape.shape[0]):
        if links_to_scrape["n_films"][link] > 200:
          up_to = 202
        else:
          up_to = links_to_scrape["n_films"][link]
        for i in tqdm.tqdm(range(1, int(up_to), 50)):
            url = f'{links_to_scrape["link"][link]}&start={i}'
            page = requests.get(url)
            soup = BeautifulSoup(page.content, features="lxml")
            one_page = soup.find_all('div', class_='lister-item-content')
            tmp_df = scrape_info_from_one_page(one_page)
            tmp_df['page_url'] = url
            all_films = all_films.append(tmp_df, ignore_index=True)
            time.sleep(np.random.uniform())
        
    return all_films


def scrape_n_films_for_each_date(release_dates):
    n_films_for_each_link = pd.DataFrame()
    for date in tqdm.tqdm(range(0, len(release_dates)-1, 1)):
        url = f'https://www.imdb.com/search/title/?release_date={release_dates[date]},{release_dates[date+1]}'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, features="lxml")

        n_films = soup.find('div', class_='desc').find('span').text
        n_films = re.sub('1-50 of ', '', n_films)
        n_films = re.sub(' titles.', '', n_films)
        n_films = int(re.sub(',', '', n_films))
        n_films_for_each_link = n_films_for_each_link.append({
            'link': url, 'n_films': n_films
        }, ignore_index=True)
    return n_films_for_each_link