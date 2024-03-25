import pytest
import os

import sys
sys.path.append('../../tinytroupe/')
sys.path.append('../../')
sys.path.append('..')


from tinytroupe.examples import create_oscar_the_architect
from tinytroupe.control import Simulation
import tinytroupe.control as control
from tinytroupe.personfactory import TinyPersonFactory
from tinytroupe.personchecker import TinyPersonChecker

from testing_utils import *

def test_validate_person(setup):

    ##########################
    # Banker
    ##########################
    banker_spec =\
    """
    A vice-president of one of the largest brazillian banks. Has a degree in engineering and an MBA in finance. 
    Is facing a lot of pressure from the board of directors to fight off the competition from the fintechs.    
    """
    banker_factory = TinyPersonFactory(banker_spec)
    banker = banker_factory.generate_person()
    banker_expectations =\
    """
    He/she is:
    - Wealthy
    - Very intelligent and ambitious
    - Has a lot of connections
    - Is in his 40s or 50s

    Tastes:
    - Likes to travel to other countries
    - Either read books, collect art or play golf
    - Enjoy only the best, most expensive, wines and food
    - Dislikes communists, unions and the like

    Other notable traits:
    - Has some stress issues, and might be a bit of a workaholic
    - Deep knowledge of finance, economics and financial technology
    - Is a bit of a snob
    - Might pretend to be a hard-core woke, but in reality that's just a facade to climb the corporate ladder  
    """
    banker_score, banker_justification = TinyPersonChecker.validate_person(banker, expectations=banker_expectations, include_agent_spec=False, max_content_length=None)

    assert banker_score > 0.5, f"Validation score is too low: {banker_score:.2f}"


    ##########################
    # Busy Knowledge Worker   
    ########################## 
    bkw_spec =\
    """
    A typical knowledge worker in a large corporation grinding his way into upper middle class.
    """
    bkw_factory = TinyPersonFactory(bkw_spec)
    busy_knowledge_worker = bkw_factory.generate_person()
    bkw_expectations =\
    """
    Some characteristics of this person:
    - Very busy
    - Likes to have lunch with colleagues
    - To travel during vacations
    - Is married and worrying about the cost of living, particularly regarding his/her children
    - Has some stress issues, and potentially some psychiatric problems
    - Went to college and has a degree in some technical field
    - Has some very specific skills
    - Does not have a wide range of interests, being more focused on his/her career, family and very few hobbies if any
    """

    bkw_score, bkw_justification = TinyPersonChecker.validate_person(busy_knowledge_worker, expectations=bkw_expectations, include_agent_spec=False, max_content_length=None)

    assert bkw_score > 0.5, f"Validation score is too low: {bkw_score:.2f}"

    # Now, let's check the score for the busy knowledge worker with the wrong expectations! It has to be low!
    wrong_expectations_score, wrong_expectations_justification = TinyPersonChecker.validate_person(busy_knowledge_worker, expectations=banker_expectations, include_agent_spec=False, max_content_length=None)

    assert wrong_expectations_score < 0.5, f"Validation score is too high: {wrong_expectations_score:.2f}"