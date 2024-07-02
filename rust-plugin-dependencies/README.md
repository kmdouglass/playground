# Rust Plugins with Dependencies

Prototype of a plugin system where plugins can depend on other plugins.

## Notes

I use the `Any` trait to downcast the `Plugin` trait objects to their concrete types, which gives me access to specific methods implemented on those types. See the`Plugin2::do_something_special` method and its use in `main()`.
