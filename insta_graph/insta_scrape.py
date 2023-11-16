import json
import random
import time
from typing import List, Dict
import os
from datetime import datetime

from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

MUTUALS_MODAL_CSS_ELEMENT = ".x1lliihq.x1plvlek.xryxfnj.x1n2onr6.x193iq5w.xeuugli.x1fj9vlw.x13faqbe.x1vvkbs.x1s928wv.xhkezso.x1gmr53x.x1cpjm7i.x1fgarty.x1943h6x.x1i0vuye.x1fhwpqd.xk50ysn.x1roi4f4.x1s3etm8.x676frb.x10wh9bi.x1wdrske.x8viiok.x18hxmgj"
ALL_FOLLOWERS_MODAL_CSS_ELEMENT = ".x1i10hfl.xjqpnuy.xa49m3k.xqeqjp1.x2hbi6w.xdl72j9.x2lah0s.xe8uvvx.xdj266r.x11i5rnm.xat24cr.x1mh8g0r.x2lwn1j.xeuugli.x1hl2dhg.xggy1nq.x1ja2u2z.x1t137rt.x1q0g3np.x1lku1pv.x1a2a7pz.x6s0dn4.xjyslct.x1ejq31n.xd10rxx.x1sy0etr.x17r0tee.x9f619.x1ypdohk.x1i0vuye.x1f6kntn.xwhw2v2.xl56j7k.x17ydfre.x2b8uid.xlyipyv.x87ps6o.x14atkfc.xcdnw81.xjbqb8w.xm3z3ea.x1x8b98j.x131883w.x16mih1h.x972fbf.xcfux6l.x1qhh985.xm0m39n.xt0psk2.xt7dq6l.xexx8yu.x4uap5.x18d9i69.xkhd6sd.x1n2onr6.x1n5bzlp.x173jzuc.x1yc6y37.xjypj1w"
FOLLOWER_ROW_CSS_ELEMENT = '.x1dm5mii.x16mil14.xiojian.x1yutycm.x1lliihq.x193iq5w.xh8yej3'


def transform_ig_provided_following_list(input_filename, output_filename):
    """
    Transform the 'following.json' file format to a suitable format for loading scraped connections.

    This function reads the Instagram 'following.json' file, processes the contained relationships,
    and outputs a new JSON file with a simplified structure that is convenient for loading
    scraped connections.

    Parameters:
    - input_filename : str
        The path to the original 'following.json' file received from Instagram.
    - output_filename : str
        The path where the transformed data should be saved.

    The output JSON file will have the following structure:
    [
        {
            "username": "example_username",
            "timestamp": 1589642123
        },
        ...
    ]

    The function will exit early if the output file already exists to prevent overwriting.

    Returns:
    None
    """
    # Check if the output file already exists
    if os.path.isfile(output_filename):
        print(f"The file '{output_filename}' already exists.")
        return  # Exit the function early

    # Open and read the input JSON file
    with open(input_filename, 'r') as file:
        data = json.load(file)

    # Process the data to match the desired structure
    transformed_data = []
    for relationship in data['relationships_following']:
        for string_data in relationship['string_list_data']:
            transformed_data.append({
                "username": string_data['value'],
                "timestamp": string_data['timestamp']
            })

    # Write the transformed data to the output JSON file
    with open(output_filename, 'w') as file:
        json.dump(transformed_data, file, indent=4)
        print(f"Data has been written to '{output_filename}' successfully.")


def extract_text_from_html(html_content: str) -> list:
    """
    Extracts all non-empty text strings from an HTML content.

    Parameters:
    - html_content (str): A string of the HTML content.

    Returns:
    - list: A list of extracted non-empty text strings.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    texts = [text for text in soup.stripped_strings]  # stripped_strings is an iterator that extracts non-empty strings
    return texts


def get_to_modal(driver) -> None:
    """
    Navigates and clicks elements in a web page to reach a modal.
    If any element is not found, it will return early.


    Returns:
    - None: If successful or if there's a NoSuchElementException.
    """
    try:
        time.sleep(random.uniform(0.4,1.2))
        modal_launch = driver.find_element(By.CSS_SELECTOR, MUTUALS_MODAL_CSS_ELEMENT)
        modal_launch.click()
        time.sleep(random.uniform(0.4,1.2))
        all_followers_launch = driver.find_element(By.CSS_SELECTOR, ALL_FOLLOWERS_MODAL_CSS_ELEMENT)
        all_followers_launch.click()
        return

    except NoSuchElementException:
        return 0



from selenium.webdriver.common.by import By


def get_following_list(json_path='following_data.json'):
    with open(json_path, 'r') as f:
        following_data = json.load(f)
    following = []
    for user in following_data:
        following.append(user['username'])
    return following

def cache_user_data(driver, user_data_cache: list, json_path = 'following_data.json') -> list:
    """
    Extracts user data from web page elements based on the provided CSS selector and
    caches the data into a list if it's not already present.

    Note: This function assumes the presence of the global driver object from Selenium.

    Parameters:
    - user_data_cache (list): The existing list to which new user data will be appended.

    Returns:
    - list: An updated list containing user data dictionaries.
    """

    child_elements = driver.find_elements(By.CSS_SELECTOR, FOLLOWER_ROW_CSS_ELEMENT)
    for element in child_elements:
        data = extract_text_from_html(element.get_attribute('outerHTML'))
        if (data[0] != 'nboveri'): #There will always be a username and a follow button, but not always name
            if len(data) == 3:
                user_data = {
                    'username' : data[0],
                    'name' : data[1],
                    'follow_status': data[2] #'Following' if data[0] in following else 'Follow' # We can't trust Instagram's assessment of whether we're following or not
                }
            else:
                user_data = {
                    'username' : data[0],
                    'name' : "",
                    'follow_status': data[1] #'Following' if data[0] in following else 'Follow'
                }

            if user_data not in user_data_cache:
                user_data_cache.append(user_data)
    return user_data_cache



def count_following(user_data: List[Dict[str, str]]) -> int:
    """
    Counts the number of users with a 'Following' status from a list of user data.

    Parameters:
    - user_data (List[Dict[str, str]]): A list of dictionaries where each dictionary
      represents user data, with a key named 'follow_status' that indicates the user's follow status.

    Returns:
    - int: The number of users with a 'Following' status.
    """
    count = 0
    if user_data:
        for user in user_data:
            if user['follow_status'] == 'Following':
                count += 1
    return count

def scroll_smoothly(driver, element, step: int = 100, delay: float = 0.05) -> None:
    """
    Scroll an element smoothly in a browser using a specified step and delay.

    Parameters:
    - driver: Selenium webdriver instance.
    - element: The web element to scroll.
    - step (int, optional): Number of pixels to scroll each time. Defaults to 100.
    - delay (float, optional): Delay between each scroll step in seconds. Defaults to 0.05.

    Returns:
    - None: The function doesn't return any value; it performs a side-effect of scrolling an element.
    """
    current_scroll_position, new_scroll_position = 0, 0

    # Get the scroll height of the element
    scroll_height = driver.execute_script("return arguments[0].scrollHeight", element)

    while current_scroll_position < scroll_height:
        current_scroll_position = new_scroll_position

        # Scroll down incrementally
        driver.execute_script(f"arguments[0].scrollTop = arguments[0].scrollTop + {step}", element)

        # Pause for a moment
        time.sleep(delay)

        # Fetch the new scroll position
        new_scroll_position = driver.execute_script("return arguments[0].scrollTop", element)

        # If the scroll position no longer changes, break out of the loop
        if new_scroll_position == current_scroll_position:
            break



from selenium.common.exceptions import TimeoutException
from typing import List, Dict
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random

class InstagramNotLoadingModalContent(Exception):
    def __init__(self, username):
        self.username = username
        self.timestamp = datetime.now()
        message = f"Modal did not load any mutual follower content for user '{username}' at {self.timestamp}."
        super().__init__(message)

def extract_mutual_followers(driver, username: str, json_path='following_data.json') -> List[Dict[str, str]]:
    """
    Extracts the list of mutual followers for a given Instagram username.

    This function navigates to the Instagram profile of the specified user, opens
    the modal showing followers, and scrolls through it to extract mutual followers' data.

    Note: This function assumes the presence of the global driver object from Selenium.

    Parameters:
    - username (str): The Instagram username for which mutual followers need to be extracted.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries where each dictionary represents user data.
    """
    driver.get(f"https://www.instagram.com/{username}")
    time.sleep(random.uniform(2.5, 3))

    try:
        WebDriverWait(driver, 3).until(lambda driver: get_to_modal(driver) != 0)
    except:
        if "Sorry, this page isn't available." in extract_text_from_html(driver.page_source):
            return ["Load Issue"]
        else:
            return []

    modal_scrollable = driver.find_element(By.CLASS_NAME, '_aano')

    # Wait for the modal and initial child elements to load
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, FOLLOWER_ROW_CSS_ELEMENT)))

    all_user_data = []

    # Check if there's an initial list of mutual followers
    initial_count = count_following(cache_user_data(driver, all_user_data, json_path=json_path))
    if initial_count == 0:
        raise InstagramNotLoadingModalContent("The modal did not load any mutual follower content.")

    while True:
        following_before_scroll = initial_count
        time.sleep(random.uniform(0.2, 0.4))

        scroll_smoothly(driver, modal_scrollable)
        try:
            # Waiting for a new element to load. It will throw a TimeoutException if no new elements are loaded in 2 seconds.
            WebDriverWait(driver, 5).until(
                lambda driver: count_following(cache_user_data(driver, all_user_data, json_path=json_path)) > following_before_scroll)
        except TimeoutException:
            # No new followers were loaded after scroll
            break

    return all_user_data



def update_mutual_followers_for_users(driver, json_filepath: str, user_list: List[str], output_filepath: str):
    """
    Updates mutual followers for a list of usernames and saves the results in a new JSON file.

    Parameters:
    - driver: Selenium WebDriver instance
    - json_filepath (str): Path to the JSON file containing initial user data.
    - user_list (List[str]): List of Instagram usernames to be updated.
    - output_filepath (str): Path where the updated JSON data will be saved.

    Returns:
    - None
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    users_to_update = [user for user in data if ('mutual_followers' in user) and (user['username'] in user_list)]
    for user in tqdm(users_to_update):
        user['mutual_followers'] = extract_mutual_followers(driver, user['username'])
        with open(output_filepath, 'w') as f:
            json.dump(data, f, indent=4)


def load_all_mutual_followers(driver, json_path: str, start_element: str = None, count_element: int = 85):
    """
    Updates the mutual followers for a given list of users in a JSON file.

    Parameters:
    - json_path (str): Path to the JSON file containing user data.
    - start_element (str, optional): Username from which to start updating mutual followers. Defaults to None.
    - count_element (int, optional): Number of users to update. Defaults to 85.

    Returns:
    - None
    """
    with open(json_path, 'r') as f:
        following_data = json.load(f)

    start_now = True if start_element is None else False

    count = 0
    for user in tqdm(following_data):
        # If start_element is provided, look for it and start from there
        if start_element and user['username'] == start_element:
            start_now = True

        if not start_now:
            continue

        # Limit the number of names to process
        count += 1
        if count > count_element:
            break

        # Check if 'mutual_followers' key exists and its list is empty
        if user.get('mutual_followers') == []:
            user['mutual_followers'] = extract_mutual_followers(driver, user['username'])
        elif 'mutual_followers' not in user:
            user['mutual_followers'] = extract_mutual_followers(driver, user['username'])

        # Write updated data back to JSON file
        with open(json_path, 'w') as f:
            json.dump(following_data, f, indent=4)

# Make sure to define or import the extract_mutual_followers function before calling load_all_mutual_followers.



def last_populated_following(user_data: List[Dict[str, str]]) -> str:
    """
    Determines the username of the first unpopulated follower after the last populated one.

    Parameters:
    - user_data (List[Dict[str, str]]): A list of user dictionaries containing 'username'
      and potentially 'mutual_followers' keys.

    Returns:
    - str: Returns the username of the first unpopulated follower or "Load complete" if
      all users are populated.
    """
    # Initialize the index for the last populated element.
    last_populated_index = None

    # Loop through each user in the user data.
    for i, user in enumerate(user_data):
        # Check if 'mutual_followers' exists and is not a load issue.
        mutual_followers = user.get('mutual_followers', None)
        if mutual_followers is not None and mutual_followers != ['Load Issue']:
            # Update the last populated index.
            last_populated_index = i

    # If the last populated index is the last in the list, return "Load complete".
    if last_populated_index is not None and last_populated_index == len(user_data) - 1:
        return 'Load complete'

    # If there's no populated data or the next one is unpopulated, return the next username.
    if last_populated_index is None:
        return 'No populated users found'
    else:
        return user_data[last_populated_index + 1]['username']


