from __future__ import print_function


class Cache(dict):
    def invalidate(self, period, entity_name, variable=None):
        """
        Invalidates all keys matching period, entity_name and possibly variable

        if variable is None, it matches all keys for that period and entity
        """
        # print("invalidate", variable.name, period, entity_name)
        for i, key in enumerate(self.keys()):
            # print(key)
            c_expr, c_period, c_entity_name, c_filter_expr = key
            # XXX: do we also need to invalidate when name not in expr but
            # name in filter_expr?
            expr_match = variable is None or variable in c_expr
            if (c_period == period and c_entity_name == entity_name and
                    expr_match):
                # print("matches", key, " => invalidating")
                del self[key]

