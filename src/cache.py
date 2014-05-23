from __future__ import print_function


class Cache(dict):
    def invalidate(self, variable, period, entity_name):
        # print("invalidate", variable.name, period, entity_name)
        for i, key in enumerate(self.keys()):
            # print(key)
            c_expr, c_period, c_entity_name, c_filter_expr = key
            #XXX: do we also need to invalidate when name not in expr but
            # name in filter_expr?
            #TODO: variable should contain entity_name
            if (c_period == period and c_entity_name == entity_name and
                    variable in c_expr):
                # print("matches", key, " => invalidating")
                del self[key]
