# Developer tools useful for maintaining the repository

## Code headers

The code headers can be standardized by running

``` bash
python dev-tools/update_license_headers.py
```

(you need )

from the repository root.

You can edit the `NOTICE.yml` to update the header.

## Docs

```
pdoc --logo https://i.ibb.co/NstG5hz/03.png -o html_docs dlc2action
```

## Codeformatting

```
black .
```

## Testing

We use pytests for testing. Simply run:

```
pytest
```
