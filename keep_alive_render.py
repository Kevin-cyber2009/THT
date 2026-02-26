#!/usr/bin/env python3
# keep_alive_render.py
"""
Keep Render API alive - no 50s cold start!

USAGE:
    python keep_alive_render.py --url https://your-app.onrender.com
    
    # Run in background (Windows)
    pythonw keep_alive_render.py --url https://your-app.onrender.com
"""

import requests
import time
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ping_server(url):
    """Ping server to keep it alive"""
    try:
        start = time.time()
        response = requests.get(f"{url}/health", timeout=30)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            logger.info(f"✅ Ping OK - {elapsed:.2f}s - Status: {response.status_code}")
            return True
        else:
            logger.warning(f"⚠️  Ping failed - Status: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Ping error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Keep Render API alive')
    parser.add_argument('--url', required=True, help='API URL (e.g., https://your-app.onrender.com)')
    parser.add_argument('--interval', type=int, default=600, help='Ping interval in seconds (default: 600 = 10 min)')
    
    args = parser.parse_args()
    
    url = args.url.rstrip('/')
    interval = args.interval
    
    logger.info("="*60)
    logger.info("RENDER KEEP-ALIVE SERVICE")
    logger.info("="*60)
    logger.info(f"URL: {url}")
    logger.info(f"Interval: {interval}s ({interval/60:.1f} minutes)")
    logger.info(f"Started: {datetime.now()}")
    logger.info("="*60)
    logger.info("")
    
    # Initial ping
    ping_server(url)
    
    # Loop
    try:
        while True:
            time.sleep(interval)
            ping_server(url)
    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("Service stopped")
        logger.info("="*60)


if __name__ == '__main__':
    main()
