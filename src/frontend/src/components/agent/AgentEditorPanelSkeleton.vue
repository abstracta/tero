<template>
  <div class="flex flex-col gap-3 px-4 py-2 mb-10 overflow-y-auto animate-pulse min-h-0">

    <!-- Basic Info section header -->
    <div class="flex justify-between items-center border-b pb-3 mb-2">
      <div class="h-4 bg-auxiliar-gray rounded w-36"></div>
      <div class="h-5 w-5 bg-auxiliar-gray rounded"></div>
    </div>

    <!-- Icon + Visibility -->
    <div class="flex flex-row justify-between items-center">
      <div class="w-12 h-12 bg-auxiliar-gray rounded-full"></div>
      <div class="flex flex-col gap-2 w-7/12">
        <div class="h-4 bg-auxiliar-gray rounded w-32"></div>
        <div class="h-10 bg-auxiliar-gray rounded w-full"></div>
      </div>
    </div>

    <!-- Name -->
    <div class="form-field gap-2">
      <div class="h-4 bg-auxiliar-gray rounded w-18"></div>
      <div class="h-10 bg-auxiliar-gray rounded w-full"></div>
      <div class="h-3 bg-auxiliar-gray rounded w-64"></div>
    </div>

    <!-- Description -->
    <div class="form-field gap-2">
      <div class="h-4 bg-auxiliar-gray rounded w-24"></div>
      <div class="h-10 bg-auxiliar-gray rounded w-full"></div>
      <div class="h-3 bg-auxiliar-gray rounded w-56"></div>
    </div>

    <!-- Behavior section header -->
    <div class="flex justify-between items-center border-b pb-3 mb-2 mt-4">
      <div class="h-4 bg-auxiliar-gray rounded w-40"></div>
      <div class="h-5 w-5 bg-auxiliar-gray rounded"></div>
    </div>

    <!-- Model + Reasoning -->
    <div class="flex flex-row gap-3">
      <div class="form-field w-full">
        <div class="h-4 bg-auxiliar-gray rounded w-20 mb-2"></div>
        <div class="h-10 bg-auxiliar-gray rounded w-52"></div>
      </div>
      <div class="form-field">
        <div class="h-4 bg-auxiliar-gray rounded w-28 mb-2"></div>
        <div class="h-10 bg-auxiliar-gray rounded w-52"></div>
      </div>
    </div>

    <!-- Instructions -->
    <div class="form-field gap-2">
      <div class="h-4 bg-auxiliar-gray rounded w-28"></div>
      <div class="h-40 bg-auxiliar-gray rounded w-full"></div>
      <div class="h-3 bg-auxiliar-gray rounded w-72"></div>
    </div>

    <!-- Conversation starters -->
    <div class="form-field gap-2">
      <div class="h-4 bg-auxiliar-gray rounded w-40"></div>
      <div class="h-20 bg-auxiliar-gray rounded w-full"></div>
    </div>

    <!-- Tools -->
    <div class="form-field gap-2">
      <div class="h-4 bg-auxiliar-gray rounded w-20"></div>
      <div class="h-20 bg-auxiliar-gray rounded w-full"></div>
    </div>
  </div>
</template>
