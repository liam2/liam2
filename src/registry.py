from __future__ import print_function


class EntityRegistry(dict):
    def add(self, entity):
        self[entity.name] = entity

entity_registry = EntityRegistry()
