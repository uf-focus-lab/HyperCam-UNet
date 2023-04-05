# Author: Yuxuan Zhang
"""Name | Bandwidth(nm) | Delta Bandwidth(nm)"""
LED_LIST = [
	("U-Violet", 395, 10),
	("Blue"    , 466, 15),
	("Green"   , 520, 15),
	("Y-Green" , 573, 20),
	("Yellow"  , 585, 20),
	("Orange"  , 600, 20),
	("Red"     , 660, 17),
	("Infrared", 940, 40),
]

REF_BANDS = list(range(402, 998 + 2, 2))
