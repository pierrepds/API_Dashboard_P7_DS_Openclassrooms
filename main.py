# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:56:03 2022

@author: Pierre
"""

# Import useful package
from fastapi import FastAPI

# Creation of the API
app = FastAPI()

# Root route
@app.get("/")
async def root():
    return{'message':'API Credit Scoring'}