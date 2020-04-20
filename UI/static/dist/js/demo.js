/**
 * AdminLTE Demo Menu
 * ------------------
 * You should not use this file in production.
 * This file is for demo purposes only.
 */
(function ($) {
  'use strict'

  var $sidebar   = $('.control-sidebar')
  var $container = $('<div />', {
    class: 'p-3 control-sidebar-content'
  })

  $sidebar.append($container)

  $container.append(
    '<h5>Driver Management GUI - Help!</h5><hr class="mb-2"/>'
  )

  $container.append('<div>If you need help, please <a href ="">contact with us</a>.</div>')

})(jQuery)