import sys
sys.path.append("")

import json
import logging
import requests
from bs4 import BeautifulSoup
import re
import time
from pathlib import Path
import pandas as pd  # Import pandas for saving DataFrame to CSV
from src.utils import check_path  # Ensure you import the function to check folder path

# Cấu hình logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

THREAD_CACHE_DIR = Path("cache")  # Specify your cache directory here
CSV_OUTPUT_DIR = Path("output")  # Directory to save CSV files
VOZ_BASE_URL = "https://voz.vn"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://voz.vn/",
}


class ThreadCrawler:
    """
    Class để crawl một thread cụ thể từ VOZ, hỗ trợ phân trang
    """
    def get_thread(self, thread_url, use_cache=True, max_pages=None, save_data=False):
        """
        Lấy dữ liệu của một thread cụ thể, hỗ trợ phân trang

        Args:
            thread_url (str): URL của thread cần crawl
            use_cache (bool): Sử dụng cache nếu có
            max_pages (int, optional): Số trang tối đa cần crawl, None để crawl tất cả
            save_data (bool): Lưu dữ liệu vào CSV nếu True

        Returns:
            dict: Dữ liệu của thread
        """
        # Trích xuất thread_id từ URL
        thread_id = self._extract_thread_id(thread_url)
        if not thread_id:
            logger.error(f"Không thể trích xuất thread ID từ URL: {thread_url}")
            return None
            
        # Kiểm tra cache
        cache_file = THREAD_CACHE_DIR / f"{thread_id}.json"
        if use_cache and cache_file.exists():
            logger.info(f"Sử dụng cache cho thread {thread_id}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    thread_data = json.load(f)
                    # Kiểm tra xem cache có đầy đủ không
                    if max_pages is None or len(thread_data.get('pages', [])) >= max_pages:
                        return thread_data
                    logger.info(f"Cache không đầy đủ, tiếp tục crawl từ trang {len(thread_data.get('pages', []))+1}")
            except Exception as e:
                logger.error(f"Lỗi khi đọc cache: {str(e)}")

        # Crawl thread
        thread_data = self._crawl_thread(thread_url, thread_id, max_pages)
        if thread_data:
            # Lưu vào cache
            self._save_cache(thread_data, cache_file)

            if save_data:
                self._save_to_csv(thread_data, thread_id)  # Save post data to CSV
            
        return thread_data
    
    def _extract_thread_id(self, thread_url):
        """Trích xuất thread ID từ URL"""
        match = re.search(r'\.(\d+)/?', thread_url)
        return match.group(1) if match else None
    
    def _crawl_thread(self, thread_url, thread_id, max_pages=None):
        """
        Crawl một thread cụ thể với hỗ trợ phân trang

        Args:
            thread_url (str): URL của thread
            thread_id (str): ID của thread
            max_pages (int, optional): Số trang tối đa cần crawl, None để crawl tất cả

        Returns:
            dict: Dữ liệu của thread
        """
        try:
            # Truy cập vào thread
            response = requests.get(thread_url, headers=HEADERS)
            if response.status_code != 200:
                logger.error(f"Không thể tải thread: {thread_url}")
                return None
            
            # Lấy HTML của trang
            page_source = response.content
            
            # Parse HTML
            soup = BeautifulSoup(page_source, 'lxml')

            # Lấy tiêu đề thread
            title_element = soup.select_one('.p-title-value')
            thread_title = title_element.text.strip() if title_element else ""
            
            # Kiểm tra số trang
            current_page = 1
            total_pages = 1
            
            # Tìm phân trang
            # VOZ uses XenForo — pagination is in nav.pageNavWrapper
            nav = soup.select_one('nav.pageNavWrapper')
            if nav:
                # Current page
                current_el = nav.select_one('.pageNav-page--current')
                if current_el:
                    current_page = int(current_el.text.strip())

                # Last numbered page link gives total pages
                page_links = nav.select('.pageNav-page a')
                page_nums = [int(a.text.strip()) for a in page_links if a.text.strip().isdigit()]
                if page_nums:
                    total_pages = max(page_nums)
            
            logger.info(f"Thread {thread_id} có {total_pages} trang")
            
            if max_pages is not None:
                total_pages = min(total_pages, max_pages)
                logger.info(f"Giới hạn crawl {total_pages} trang")
            
            # Crawl từng trang
            thread_data = {
                "thread_id": thread_id,
                "title": thread_title,
                "url": thread_url,
                "crawl_date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_pages": total_pages,
                "pages": []
            }
            
            # Crawl trang đầu tiên (đã load rồi)
            page_data = {
                "page_number": current_page,
                "posts": self._parse_posts(soup)
            }
            thread_data["pages"].append(page_data)
            logger.info(f"Đã crawl trang {current_page}/{total_pages}")
            
            # Crawl các trang tiếp theo
            for page_num in range(2, total_pages + 1):
                next_page_url = f"{thread_url.rstrip('/')}/page-{page_num}" # Fix the URL format
                page_data = self._crawl_page(next_page_url, page_num)
                if page_data:
                    thread_data["pages"].append(page_data)
                    logger.info(f"Đã crawl trang {page_num}/{total_pages}")
                else:
                    logger.error(f"Không thể crawl trang {page_num}")
                    break
            
            # Tạo danh sách tất cả các posts từ tất cả các trang
            all_posts = []
            for page in thread_data["pages"]:
                all_posts.extend(page["posts"])
            
            thread_data["posts"] = all_posts
            thread_data["post_count"] = len(all_posts)
            
            return thread_data
            
        except Exception as e:
            logger.error(f"Lỗi khi crawl thread {thread_url}: {str(e)}")
            return None
    
    def _crawl_page(self, page_url, page_num):
        """Crawl một trang cụ thể của thread"""
        # try:
        # time.sleep(3)  # Thêm delay để tránh bị block
        response = requests.get(page_url, headers=HEADERS)
        if response.status_code != 200:
            logger.error(f"Không thể tải trang: {page_url}")
            logger.info("response", response)
            return None
        
        page_source = response.content
        soup = BeautifulSoup(page_source, 'lxml')
        posts = self._parse_posts(soup)
        
        return {
            "page_number": page_num,
            "posts": posts
        }
            
        # except Exception as e:
        #     logger.error(f"Lỗi khi crawl trang {page_url}: {str(e)}")
        #     return None
            
        
    def _save_cache(self, thread_data, cache_file):
        """Lưu dữ liệu thread vào cache"""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(thread_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu cache cho thread {thread_data['thread_id']}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu cache: {str(e)}")
            return False
      
    def _parse_posts(self, soup):
        """Parse các bài viết từ HTML"""
        posts = []
        post_elements = soup.select('.block-body article.message')
        
        for element in post_elements:
            post_id = element.get('data-content', '').replace('post-', '') if 'data-content' in element.attrs else None
            username_element = element.select_one('.message-userDetails .username')
            username = username_element.text.strip() if username_element else "Unknown"
            user_id = username_element.get('data-user-id') if username_element else None
            time_element = element.select_one('.message-attribution-main time')
            created_date = time_element.get('datetime', '') if time_element else ""
            modified_date = element.select_one('.message-lastEdit time').get('datetime', '') if element.select_one('.message-lastEdit time') else None
            content_element = element.select_one('.message-body .bbWrapper')
            content = content_element.text.strip() if content_element else ""
            
            # Remove "Click to expand..." from content
            content = content.replace("Click to expand...", "").strip()

            posts.append({
                "post_id": post_id,
                "author_username": username,  # Change to separate columns
                "author_user_id": user_id,     # Change to separate columns
                "created_date": created_date,
                "modified_date": modified_date,
                "content_text": content.replace('\n', ' ').replace('\r', '')  # Replace line breaks with spaces
            })
        
        return posts

    def _save_to_csv(self, thread_data, thread_id):
        """
        Lưu dữ liệu bài viết vào file CSV.

        Args:
            thread_data (dict): Dữ liệu chứa thông tin bài viết.
            thread_id (str): ID của thread.
        """
        check_path(CSV_OUTPUT_DIR)  # Ensure the output directory exists
        
        posts = thread_data.get('posts', [])
        if posts:
            df = pd.DataFrame(posts)  # Create DataFrame from posts data
            csv_file_path = CSV_OUTPUT_DIR / f"thread_{thread_id}_posts.csv"  # Name the CSV file
            df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')  # Save DataFrame to CSV
            logger.info(f"Đã lưu bài viết vào file CSV: {csv_file_path}")
        else:
            logger.warning("Không có bài viết nào để lưu vào CSV.")

if __name__ == "__main__":
    # Test crawl một thread cụ thể
    test_url = "https://voz.vn/t/ke-ve-nhung-niem-vui-rat-doi-thuong-ma-da-lau-ban-khong-lam-nua.1228919/"
    
    crawler = ThreadCrawler()
    try:
        thread_data = crawler.get_thread(test_url, use_cache=False, max_pages=None, save_data=True)
        if thread_data:
            print(f"Đã crawl thread: {thread_data['title']}")
            print(f"Số lượng bài viết: {thread_data['post_count']}")
    finally:
        print("Crawl complete.")