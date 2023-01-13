from os.path import exists
from pathlib import Path
from uuid import uuid4
def uniq(path, ext="", l=8):
	path = Path(path)
	while True:
		ID = str(uuid4()).replace('-', '')[:l]
		filename = ID + ext
		if not exists(path / filename):
			return ID
