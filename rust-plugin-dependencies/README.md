# Rust Plugins with Dependencies

Prototype of a plugin system where plugins can depend on other plugins.

## Notes

I use the `Any` trait to downcast the `Plugin` trait objects to their concrete types, which gives me access to specific methods implemented on those types. See the`Plugin2::do_something_special` method and its use in `main()`.

The plugin system is implemented as a Hashmap, and Plugins provide their own dependencies. The dependencies are added to the HashMap if they don't already exist. Moving a Plugin and its dependencies into the HashMap must be done in the right order for the code to compile.
