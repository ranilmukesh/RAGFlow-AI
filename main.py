import streamlit as st
from rag_ui import RAGUI
from imports import *
import multiprocessing
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    config = {
        "cache": {
            "path": "cache/cache.db",
            "max_size_mb": 500,
            "ttl": 3600,
            "cleanup_interval": 3600
        },
        "search": {
            "initial_results_multiplier": 3,
            "top_k": 5
        },
        "processing": {
            "batch_size": 64,
            "max_concurrent": min(multiprocessing.cpu_count() * 2, 8)
        },
        "temp_directory": str(Path("temp").absolute())
    }
    
    ui = RAGUI(config)
    ui.main()