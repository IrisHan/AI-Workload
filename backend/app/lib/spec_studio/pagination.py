"""Pure instance pagination transplanted from Spec Studio api/snapshots.py."""


def instance_page(result, offset, limit):
    total = result["backbone"]["num_layers"]
    items = []
    for segment in result["execution"]:
        start = max(offset, segment["start"])
        stop = min(offset + limit, segment["stop_exclusive"])
        group = next(group for group in result["groups"] if group["id"] == segment["group_id"])
        for index in range(start, stop):
            items.append(dict(index=index, group_id=group["id"], template_id=group["template_id"]))
    return dict(
        items=items,
        total=total,
        offset=offset,
        limit=limit,
        next_offset=offset + limit if offset + limit < total else None,
    )
