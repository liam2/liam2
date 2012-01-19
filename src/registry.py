class EntityRegistry(dict):
    def add(self, entity):
        self[entity.name] = entity

entity_registry = EntityRegistry()
