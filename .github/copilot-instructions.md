# Copilot Instructions

1. **Keep generic cores generic.** Source-specific state, configuration, and interaction logic must live with the owning source/controller, not inside shared engines or other generic infrastructure. If only one source type needs a behavior today, expose generic engine hooks and keep the orchestration in the source layer.
